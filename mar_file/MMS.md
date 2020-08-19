## Requirements
```
pip install mxnet-model-server==1.0.4
```

## stop mxnet 
```
cd ~/Documents/M_server
conda activate M
mxnet-model-server --stop
```

## delete .mar file in mar_file folder 
```
rm -rf mar_file/*.mar *.mar
```

## build mar file 
```
model-archiver --model-name m_server --model-path ./ --handler m_emb_server:handle --export-path mar_file/
```

## run mxnet model 

```
cd mar_file/
mxnet-model-server --mms-config config.properties --model-store .
```

## call model 
open a new terminal arbitrarily

```
curl -X POST "localhost:8023/models?url=m_server.mar&batch_size=1&max_batch_delay=10&initial_workers=1"
```

```
curl -X POST http://127.0.0.1:8022/predictions/m_server -T "example.png"
```
