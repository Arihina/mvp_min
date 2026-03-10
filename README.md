SSL certificate generation
```
openssl req -x509 -newkey rsa:4096 \
-keyout key.pem \
-out cert.pem \
-days 365 \
-nodes
```

Run services in Docker
```
docker compose up -d
```
Installing a model for Ollama in Docker
```
docker exec -it ollama_server ollama pull gemma2:2b
```
or
```
docker exec -it ollama_server ollama pull gemma2:9b
```

Access to the site after the container is launched
https://127.0.0.1:8443/
