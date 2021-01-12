# lasaft-docker
This code is an adaption of [ws-choi Conditioned-Source-Separation-LaSAFT](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) to run over CPU in a Docker container.

## Build docker
```
docker build -f Dockerfile -t lasaft .
docker run --rm -it -d -v `pwd`/src:/app -v /root:/root --name lasaft-test lasaft
docker exec -it lasaft-test bash
```

## Run docker image
```
python Separation_wav.py -mix_scp /root/test.wav -yaml options/train/train.yml -model /root/best.pt -save_path ./checkpoint
```

### References
- [ws-choi Conditioned-Source-Separation-LaSAFT](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT)

