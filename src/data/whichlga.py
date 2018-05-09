import click
import os, sys
import logging
import shapefile
from shapely.geometry import shape, Point, mapping, Polygon
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv


cache = {}

def load_nsw_shapes(input_filepath, filename="2016_LGA_SHAPE/LGA_2016_AUST"):
    sf = shapefile.Reader(os.path.join(input_filepath,filename))

    nswshapes = list(map(lambda x: shape(x.__geo_interface__), sf.shapes()[:130]))
    centroids = list(map(lambda x: shape(x.__geo_interface__).centroid, sf.shapes()[:130]))

    df = pd.DataFrame(sf.records())
    nswdf = df[df[4] =="New South Wales"]
    # drop "No usual address" - not a useful LGA
    nswdf = nswdf[nswdf[0] != "LGA19499"]
    nswdf.columns = ["code", "codenum", "name", "unknown", "state", "unknown2"]
    nswdf["clean_name"] = nswdf["name"].str.extract('([^\\(]*)', expand=False).str.strip()

    lga_json = list(map(lambda x: x.__geo_interface__, sf.shapes()[:130]))

    geojson = {'type': 'FeatureCollection',
                'features': [{'type':'Feature', 'geometry':x, 'properties': {'name':nswdf.iloc[i]['clean_name']}} for i, x in enumerate(lga_json)]
              }


    return nswshapes, centroids, geojson, nswdf

def whichlga(tweetpoints, nswshapes, nswdf):
    #loop over 130 NSW LGAs and return a name or "None"
    #NB you get names from Census data which is 2016 - pre amalgamations
    output = []
    nums = []
    amalgamations = {'Botany Bay': 'Bayside',
                     'Rockdale'  : 'Bayside',
                     'Gundagai'  : 'Cootamundra-Gundagai',
                     'Western Plains Regional' : 'Dubbo Regional',
                     'Unincorporated NSW' : 'Unincorporated Far West'
                    }
    # we need to double check tweets are actually in nsw
    #rough NSW bounding box - reference https://data.gov.au/geoserver/nsw-state-boundary/wms?request=GetCapabilities
    nsw_bb = Polygon([[140.999279200001,-35.52052802079999],[140.999279200001,-28.1570199879999],
              [159.105444163417,-28.1570199879999],[159.105444163417,-37.5052802079999]])
    
    for point in tweetpoints:
        found = False
        distances = []
        if (point.x, point.y) in cache:
            output.append(cache[(point.x, point.y)])
        elif not point.within(nsw_bb):
            output.append("OutsideNSW")
            cache[(point.x, point.y)] = "OutsideNSW"
        else:
            for i, nswshape in enumerate(nswshapes):
                if point.within(nswshape):
                    found = i
                    clean_name = nswdf.iloc[found].clean_name
                    break
            if not found:
                for i, nswshape in enumerate(nswshapes):
                    distances.append(point.distance(nswshape))
                    
                #out on the water
                clean_name = nswdf.iloc[np.argmin(np.asarray(distances))].clean_name
                
            if clean_name in amalgamations:
                clean_name = amalgamations[clean_name]
            output.append(clean_name)
            cache[(point.x, point.y)] = clean_name
            
    return pd.Series(output), nums
            

def load_tweets(input_filepath, filename='tweet.csv'):
    ''' load a csv twitter dataset; trim to those with lat/lng'''
    tweetdf = pd.read_csv(os.path.join(input_filepath, "twitter_australia_loc_2018.csv"), header=0)
    loctweetdf = tweetdf.loc[tweetdf['lat'].notnull(),:]
    print("{0}/{1} tweets have lat lng".format(len(loctweetdf), len(tweetdf)))
    
    tweetpoints = []

    loctweetdf.apply(lambda x: tweetpoints.append(shape(Point(x['lng'], x['lat']))), axis=1)

    return tweetdf, tweetpoints


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default=os.path.abspath('./data/raw'))
@click.argument('output_filepath', type=click.Path(), default=os.path.abspath('./data/processed'))
def main(input_filepath, output_filepath):
    nswshapes, centroids, geojson, nswdf = load_nsw_shapes(input_filepath)

    tweetdf, tweetpoints = load_tweets(input_filepath)
    output, found = whichlga(tweetpoints, nswshapes, nswdf)

    tweetdf.loc[tweetdf['lat'].notnull(),'lga'] = np.asarray(output)

    tweetdf.to_csv(os.path.join(output_filepath,"trisma2018_w_lga.csv"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
