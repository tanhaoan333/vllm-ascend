Traceback (most recent call last):
  File "/usr/local/python3.12.13/lib/python3.12/site-packages/grpc/_server.py", line 652, in _take_response_from_response_iterator
    return next(response_iterator), True
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/d00943518/cosyvoice3/fm_multiprocess_pool/runtime/python/grpc/server.py", line 150, in Inference
    for i in model_output:
             ^^^^^^^^^^^^
  File "/home/d00943518/cosyvoice3/fm_multiprocess_pool/runtime/python/grpc/../../../cosyvoice/cli/cosyvoice.py", line 298, in inference_zero_shot_by_id
    for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/d00943518/cosyvoice3/fm_multiprocess_pool/runtime/python/grpc/../../../cosyvoice/cli/model.py", line 404, in tts
    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/d00943518/cosyvoice3/fm_multiprocess_pool/runtime/python/grpc/../../../cosyvoice/cli/model.py", line 517, in token2wav
    return self.token2wav_pool.token2wav_stream(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/d00943518/cosyvoice3/fm_multiprocess_pool/runtime/python/grpc/../../../cosyvoice/cli/process_pool.py", line 1001, in token2wav_stream
    result = self.pool.process_sync(
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/d00943518/cosyvoice3/fm_multiprocess_pool/runtime/python/grpc/../../../cosyvoice/cli/process_pool.py", line 891, in process_sync
    result = future.result(timeout=timeout)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.12.13/lib/python3.12/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.12.13/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
RuntimeError: TypeError: '>=' not supported between instances of 'NoneType' and 'tuple'
