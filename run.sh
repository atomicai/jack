docker run --name elastic_7.16.1 -v jackdb:/usr/share/elasticsearch/data -d -p 9200:9200 -e "http.host=0.0.0.0" -e "discovery.type=single-node" elasticsearch:7.16.1