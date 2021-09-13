### Build docker
        git clone -b service_tensorrt https://github.com/tonhathuy/Racing-Car-Challenge-.git
        cd Racing-Car-Challenge-
        docker-compose up
      
### [Config](./docker-compose.yml#L10)
        port 5005

### Service Response
#### HEADERS
        Content-Type: application/json
#### BODY
          {
                'code': '1000', 
                'status': rcode.code_1000, 
                'predicts_pinet': [x,y],
		'predicts_yolo': {'bboxes': boxes, 'conf': confs, 'class':clss},
                'process_time': timeit.default_timer()-start_time,
                'return': ''
          }
### Service Request

- [Base64](./test/test_predict_base64.py)
- [Binary](./test/test_predict_binary.py)
- [Binary numpy](./test/test_predict_binary_numpy.py)
- [Multi Binary](./test/test_predict_multi_binary.py)
- [Multi Part](./test/test_predict_multipart.py)
