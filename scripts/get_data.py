#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: get_data
Created: 2020-12-05

Description:

    download data from input json files

Usage:

    >>> import get_data

"""
import glob
import json
import os

import boto3


def main():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('videoblocks-ml')

    HERE = os.path.dirname(os.path.realpath(__file__))
    ROOT = os.path.dirname(HERE)
    DATA = os.path.join(ROOT, 'data')
    for jfile in glob.glob(os.path.join(DATA, 'inputs', '*.json')):
        with open(jfile, 'r') as fp:
            for line in fp:
                j = json.loads(line.strip())
                key = j['key']
                print(f"downloading {key}")
                o = bucket.Object(key)
                siid = os.path.basename(os.path.dirname(key))
                base = os.path.basename(key)
                f_local = os.path.join(DATA, f"{siid}_{base}")
                o.download_file(f_local)


if __name__ == '__main__':
    main()