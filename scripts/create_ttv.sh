#!/usr/bin/env bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -dp|--development_percentage) dp="$2"; shift ;;
        -tp|--training_percentage) tp="$2"; shift ;;
        -d|--dataset) dataset="$2"; shift ;;
        -ttv|--ttv) ttv_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Development percentage: $dp"
echo "Training percentage: $tp"
echo "Dataset: $dataset"
echo "ttv:  $ttv_dir"

create_folders() {
	[[ -d ${first_partition} ]] || mkdir -p ${first_partition}
	[[ -d ${second_partition} ]] || mkdir -p ${second_partition}
}

split() {
	calc=$(echo "scale=2; 100/$split" | bc)
	while read polyp;
	do
		result=$(echo "$nRow % $calc" | bc)
		id=$(echo ${polyp} | cut -d',' -f1)
		if (( $(echo "$result < 1.00" | bc -l) )) ; then
			grep ${id} ${metadata_file} >> ${second_partition}/metadata.csv
		else
			grep ${id} ${metadata_file} >> ${first_partition}/metadata.csv
		fi
		((nRow+=1))

	done < ${file_sort}
	while read line;
	do
	    $(echo ${line} | cut -d',' -f 2 | cut -d ' ' -f 2  >> ${first_partition_file})
	done <  ${first_partition}/metadata.csv

	while read line; do
	    $(echo ${line} | cut -d',' -f 2 | cut -d ' ' -f 2  >> ${second_partition_file})
	done < ${second_partition}/metadata.csv
}

sorted(){
	polyps=$(tail -n +2 $metadata_file | cut -d "," -f 1 | sort | uniq)
	# Looks for the different polyps and the number of images that each polyp has in metadata file
	for polyp in ${polyps}; do
		number=$(grep ${polyp} ${metadata_file} | wc -l)
		echo ${polyp},$(printf "%04d" ${number}) >> ${file_tmp}
	done 

	sort -nk 2 -t ',' -r ${file_tmp} > ${file_sort}
}

file_tmp="polyps.txt"
file_sort="sorted.txt"

# Create development and test
metadata_file=${dataset}/metadata.csv

if [[ ! -f ${metadata_file} ]]; then
	echo "Not found metadata"
	exit 1;
fi

first_partition=${ttv_dir}/development
second_partition=${ttv_dir}/test
first_partition_file=${first_partition}/development.txt
second_partition_file=${second_partition}/test.txt

nRow=0
split=$((100 - dp))

sorted
rm ${file_tmp}
create_folders
split
rm ${file_sort}

# Create train and validation in development
metadata_file=${ttv_dir}/development/metadata.csv
first_partition=${ttv_dir}/development/train
second_partition=${ttv_dir}/development/validation
first_partition_file=${first_partition}/train.txt
second_partition_file=${second_partition}/validation.txt

nRow=0
split=$((100 - tp))

sorted
rm ${file_tmp}
create_folders
split
rm ${file_sort}

