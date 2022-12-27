# Audio-Transcriber
*Version: 0.0.1*

Transcribe your .wav .mp4 .mp3 .flac files to text or record your own audio!

### Usage:
| Short Flag | Long Flag   | Description                                                   |
|------------|-------------|---------------------------------------------------------------|
| -h         | --help      | See Usage                                                     |
| -b         | --bitrate   | Bitrate to use during recording                               |
| -c         | --channels  | Number of channels to use during recording                    |
| -d         | --directory | Directory to save recording                                   |
| -f         | --file      | File to transcribe                                            |
| -m         | --model     | Model to use: <tiny, base, small, medium, large>              |
| -n         | --name      | Name of recording                                             |
| -r         | --record    | Specify number of seconds to record to record from microphone |

### Example:
```bash
audio-transcriber --file '~/Downloads/Federal_Reserve.mp4' --model 'large'
audio-transcriber --record 60 --directory '~/Downloads/' --name 'my_recording.wav' --model 'tiny'
```

#### Install Instructions
Install Python Package

```bash
python -m pip install audio-transcriber
```

##### Ubuntu Dependencies
```bash
apt install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
```

#### Build Instructions
Build Python Package

```bash
sudo chmod +x ./*.py
sudo pip install .
python3 setup.py bdist_wheel --universal
# Test Pypi
twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose -u "Username" -p "Password"
# Prod Pypi
twine upload dist/* --verbose -u "Username" -p "Password"
```
