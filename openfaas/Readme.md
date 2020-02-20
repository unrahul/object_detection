## FaaS object detection using stacks

## REST server commands

- Start the server with:

```
python rest.py 
```

- Curl the server using:

```bash
curl -i -X POST -F img=@imgs/obj1.jpg 'http://localhost:5059/detect' > img.b64
```

A base64 encoded string would be returned, to quickly view the file on terminal:

```bash
tail -n +12 img.jpg| base64 --decode > image.jpg
xgd-open image.jpg
```

There a base64_to_jpg converter function in utils.py as well.
