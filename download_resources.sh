#!/usr/bin/env bash

# for more information, see https://www.tensorflow.org/hub/common_issues

echo "Make directories"
mkdir /tmp/subnet/
mkdir /tmp/subnet/nnlm-en-dim50
mkdir /tmp/subnet/nnlm-en-dim128
mkdir /tmp/subnet/universal-sentence-encoder

echo "Download hub modules"
curl -L "https://tfhub.dev/google/nnlm-en-dim50/1?tf-hub-format=compressed" | tar -zxvC /tmp/subnet/nnlm-en-dim50
curl -L "https://tfhub.dev/google/nnlm-en-dim128/1?tf-hub-format=compressed" | tar -zxvC /tmp/subnet/nnlm-en-dim128
curl -L "https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed" | tar -zxvC /tmp/subnet/universal-sentence-encoder

echo "Upload to S3"
aws s3 cp --recursive /tmp/subnet/ s3://nucleus-chc-preprod-datasciences/users/bmcmahon/nas/tf_hub/subnet/

