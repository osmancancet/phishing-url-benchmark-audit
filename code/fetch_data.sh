#!/bin/bash
# Fetch the public Git mirrors at the audited commits and check SHA-256 of every source file.
# Usage: URLDS_ROOT=/path/to/ds ./fetch_data.sh
set -euo pipefail
DS="${URLDS_ROOT:-$HOME/ds}"; mkdir -p "$DS/probe"
[ -d "$DS/pdd" ] || git clone -q https://github.com/ebubekirbbr/pdd "$DS/pdd"
git -C "$DS/pdd" checkout -q 90b394d8f9fb5d5dc48400e8bf46b001ce42999f
[ -d "$DS/probe/GramBeddings" ] || git clone -q https://github.com/fcdalgic/GramBeddings "$DS/probe/GramBeddings"
git -C "$DS/probe/GramBeddings" checkout -q 222157e1a61a01783383d1632269ad13a7542d96
[ -d "$DS/probe/Phishing-Detection-Using-PhiUSIIL-Phishing-URL-Dataset-Website-" ] || git clone -q https://github.com/AbdulHamidNoubaa/Phishing-Detection-Using-PhiUSIIL-Phishing-URL-Dataset-Website- "$DS/probe/Phishing-Detection-Using-PhiUSIIL-Phishing-URL-Dataset-Website-"
git -C "$DS/probe/Phishing-Detection-Using-PhiUSIIL-Phishing-URL-Dataset-Website-" checkout -q 2d9a1f00a36baa33e997638ad0e8b10542278a59
[ -d "$DS/probe/Web-page-Phishing-Detection-Dataset" ] || git clone -q https://github.com/TetianaYaskova/Web-page-Phishing-Detection-Dataset "$DS/probe/Web-page-Phishing-Detection-Dataset"
git -C "$DS/probe/Web-page-Phishing-Detection-Dataset" checkout -q 9ba531ffc91985d1da652dbc4585bff057f3ed50
cd "$DS"; sha256sum -c <<SUMS
5e8fbcc88d6ba25f46bd85ed48d8aed9f56e2afc9deb712d13496112b8158ab4  pdd/input/data_legitimate_36400.json
c513d1f11b0ce159d43a23b465af2a65f752c6f65a816c2a8084f2a2adf2ebf6  pdd/input/data_phishing_37175.json
382453475373d488a4444da95c6fc856c42733150d91ec7bf10809474f57ba08  probe/GramBeddings/data/PhishStorm/valid-urls.csv
c2218f2d556ad12e59f45852a7d696157478739aba6c62e7ff0d8269904d4d96  probe/GramBeddings/data/PhishStorm/phishing-urls.csv
7e0892425caa258918ce5532c7f1249c14273ad817890350dfa9d645c4377650  probe/GramBeddings/data/ISCXURL2016/Benign_list_big_final.csv
52f7918af584f71da7f43a5cc09a5e4b612da8f8c06e34ddbd11a3736c421136  probe/GramBeddings/data/ISCXURL2016/phishing_dataset.csv
3d77bd15304205788d89bbc64f1b73ccd9163e6064183dabe58602e8bdb229f7  probe/GramBeddings/data/PDRCNN/alexa1m_dataset.csv
c76e1ec96e7efea64723b1e3b19117111a4c613a97055bc959840cdcaddad546  probe/GramBeddings/data/PDRCNN/Phishtank_dataset.csv
21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c  probe/Web-page-Phishing-Detection-Dataset/dataset_phishing.csv
a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6  probe/Phishing-Detection-Using-PhiUSIIL-Phishing-URL-Dataset-Website-/uci/PhiUSIIL_Phishing_URL_Dataset.csv
d83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01  probe/GramBeddings/data/kaggle/01-sid321axn/malicious_phish.csv
SUMS
echo DATA_OK
