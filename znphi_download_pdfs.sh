#!/bin/bash
# Author  : James Sakala
# Purpose : Download ZNPHI Reports for my Thesis Paper

ZNPHI_FILE_LINKS="http://znphi.co.zm/news/february-april-situation-reports-new-coronavirus-disease-of-2019-covid-19-sitreps/ http://znphi.co.zm/news/situation-reports-new-coronavirus-covid-19-sitreps/"
SAVE_TO_DIR="./ZNPHIPDFS"
if [[ ! -d ${SAVE_TO_DIR} ]];then mkdir -p ${SAVE_TO_DIR}; fi
pushd .
cd "$SAVE_TO_DIR"
for i in $ZNPHI_FILE_LINKS
do 
	PDF_FILE_LINKS=$(curl "http://znphi.co.zm/news/february-april-situation-reports-new-coronavirus-disease-of-2019-covid-19-sitreps/" | grep href | grep pdf | cut -d"=" -f2 |  cut -d" " -f1 | tr -d '"')
	for pdflink in $PDF_FILE_LINKS
	do 
		wget "$pdflink"; 
	done
done
popd
