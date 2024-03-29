# Audio-Transcriber

![PyPI - Version](https://img.shields.io/pypi/v/audio-transcriber)
![PyPI - Downloads](https://img.shields.io/pypi/dd/audio-transcriber)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/audio-transcriber)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/audio-transcriber)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/audio-transcriber)
![PyPI - License](https://img.shields.io/pypi/l/audio-transcriber)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/audio-transcriber)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/audio-transcriber)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/audio-transcriber)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/audio-transcriber)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/audio-transcriber)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/audio-transcriber)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/audio-transcriber)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/audio-transcriber)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/audio-transcriber)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/audio-transcriber)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/audio-transcriber)

*Version: 0.5.37*

Transcribe your .wav .mp4 .mp3 .flac files to text or record your own audio!

This repository is actively maintained - Contributions are welcome!

Contribution Opportunities:
- Support new models


<details>
  <summary><b>Usage:</b></summary>

| Short Flag | Long Flag   | Description                                                   |
|------------|-------------|---------------------------------------------------------------|
| -h         | --help      | See Usage                                                     |
| -b         | --bitrate   | Bitrate to use during recording                               |
| -c         | --channels  | Number of channels to use during recording                    |
| -d         | --directory | Directory to save recording                                   |
| -e         | --export    | Export txt, srt, and vtt files                                |
| -f         | --file      | File to transcribe                                            |
| -l         | --language  | Language to transcribe                                        |
| -m         | --model     | Model to use: <tiny, base, small, medium, large>              |
| -n         | --name      | Name of recording                                             |
| -r         | --record    | Specify number of seconds to record to record from microphone |

</details>

<details>
  <summary><b>Example:</b></summary>

```bash
audio-transcriber --file '~/Downloads/Federal_Reserve.mp4' --model 'large'
audio-transcriber --record 60 --directory '~/Downloads/' --name 'my_recording.wav' --model 'tiny'
```


</details>

<details>
  <summary><b>Model Information:</b></summary>

[Courtesy of and Credits to OpenAI: Whisper.ai](https://github.com/openai/whisper/blob/main/README.md)

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |


</details>

<details>
  <summary><b>Installation Instructions:</b></summary>

Install Python Package

```bash
python -m pip install audio-transcriber
```

##### Ubuntu Dependencies
```bash
apt install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
```

</details>

## Geniusbot Application

Use with a GUI through Geniusbot

Visit our [GitHub](https://github.com/Knuckles-Team/geniusbot) for more information

<details>
  <summary><b>Installation Instructions with Geniusbot:</b></summary>

Install Python Package

```bash
python -m pip install geniusbot
```

</details>

<details>
  <summary><b>Repository Owners:</b></summary>


<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
</details>
