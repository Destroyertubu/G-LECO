
python3 generate_data.py

wget -O books https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/5YTV8K
zstd -d books -o books_200M_uint32

wget -O fb  https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/EATHF7
zstd -d fb -o fb_200M_uint64
wget -O wiki https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/SVN8PI
zstd -d wiki -o wiki_200M_uint64
