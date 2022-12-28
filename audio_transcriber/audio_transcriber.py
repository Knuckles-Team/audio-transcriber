#!/usr/bin/env python
# coding: utf-8

import whisper
import sys
import os
import threading
import getopt
import pyaudio
import wave
import datetime
from typing import Iterator, TextIO


class AudioTranscriber:
    def __init__(self, model: str = 'base', channels: int = 1, rate: int = 44100, file_name: str = 'output.wav',
                 directory: str = os.curdir, file: str = ""):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = channels
        self.rate = rate
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.file_name = file_name
        self.title = os.path.split(self.file_name)[0]
        self.directory = directory
        self.file = None
        self.output = None
        if file != "":
            self.set_file(file)
        else:
            self.set_file(os.path.join(self.directory, self.file_name))
        self.stop = False
        self.model = whisper.load_model(model)

    def initiate_stream(self):
        self.stream = self.pyaudio_instance.open(format=self.format,
                                                 channels=self.channels,
                                                 rate=self.rate,
                                                 input=True,
                                                 frames_per_buffer=self.chunk)

    def set_file(self, file: str):
        self.file = file
        self.file_name = os.path.basename(file)
        self.directory = os.path.dirname(file)
        self.title = os.path.splitext(self.file_name)[0]
        if self.directory == "":
            self.directory = os.curdir

    def set_file_name(self, file_name: str):
        self.file_name = file_name
        self.set_file(os.path.join(self.directory, self.file_name))
        self.title = os.path.split(self.file_name)[0]

    def set_directory(self, directory: str):
        self.directory = directory
        self.set_file(os.path.join(self.directory, self.file_name))

    def set_rate(self, rate: int):
        self.rate = rate

    def set_channels(self, channels: int):
        self.channels = channels

    def set_model(self, model: str):
        if model in ['tiny', 'base', 'small', 'medium', 'large']:
            self.model = whisper.load_model(model)
        else:
            print('Model does not exist, please choose from: tiny, base, small, medium, or large')

    def record(self, seconds: int = 0):
        print("Recording started...")
        self.frames = []
        self.stop = False
        if seconds > 0:
            for i in range(0, int((self.rate / self.chunk) * seconds)):
                data = self.stream.read(self.chunk)
                self.frames.append(data)
        else:
            print("Keep recording until stop signal is sent")
            download_thread = threading.Thread(target=self.unlimited_record, name="Recorder")
            download_thread.start()
        print("Recording stopped")

    def unlimited_record(self):
        while self.stop is False:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

    def stop_stream(self):
        self.stop = True
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()

    def save_stream(self):
        wave_file = wave.open(self.file, 'wb')
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(self.pyaudio_instance.get_sample_size(self.format))
        wave_file.setframerate(self.rate)
        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

    def transcribe(self, language='en'):
        self.output = None
        start_time = datetime.datetime.now()
        print(f"Started: {start_time}\nTranscribing: {self.file}")
        self.output = self.model.transcribe(self.file, language=language)
        end_time = datetime.datetime.now()
        print(f"Ended: {end_time}\nTime Elapsed: {end_time - start_time}")
        print(f"Output: \n{self.output}")
        for segment in self.output['segments']:
            second = int(segment['start'])
            second = second - (second % 5)
            print(f'Second: {second} - Segment: \n{segment}\n\n')

    def export_text(self):
        with open(os.path.join(self.directory, f"{self.title}.txt"), "w", encoding="utf-8") as txt:
            self.write_txt(self.output["segments"], file=txt)

        with open(os.path.join(self.directory, f"{self.title}.vtt"), "w", encoding="utf-8") as vtt:
            self.write_vtt(self.output["segments"], file=vtt)

        with open(os.path.join(self.directory, f"{self.title}.srt"), "w", encoding="utf-8") as srt:
            self.write_srt(self.output["segments"], file=srt)

    @staticmethod
    def srt_format_timestamp(seconds: float):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        return f"{hours}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def write_srt(self, transcript: Iterator[dict], file: TextIO):
        count = 0
        for segment in transcript:
            count += 1
            print(
                f"{count}\n"
                f"{self.srt_format_timestamp(segment['start'])} --> {self.srt_format_timestamp(segment['end'])}\n"
                f"{segment['text'].replace('-->', '->').strip()}\n",
                file=file,
                flush=True,
            )

    @staticmethod
    def write_txt(transcript: Iterator[dict], file: TextIO):
        for segment in transcript:
            print(segment['text'].strip(), file=file, flush=True)

    def write_vtt(self, transcript: Iterator[dict], file: TextIO):
        print("WEBVTT\n", file=file)
        for segment in transcript:
            print(
                f"{self.format_timestamp(segment['start'])} --> {self.format_timestamp(segment['end'])}\n"
                f"{segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )

    @staticmethod
    def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


def usage():
    print(f"Usage: \n"
          f"-h | --help      [ See usage for script ]\n"
          f"-b | --bitrate   [ Bitrate to use during recording ]\n"
          f"-c | --channels  [ Number of channels to use during recording ]\n"
          f"-d | --directory [ Directory to save recording ]\n"
          f"-e | --export    [ Export txt, srt, & vtt ]\n"
          f"-f | --file      [ File to transcribe ]\n"
          f"-l | --language  [ Language to transcribe <'en', 'fa', 'es', 'zh'> ]\n"
          f"-m | --model     [ Model to use: <tiny, base, small, medium, large> ]\n"
          f"-n | --name      [ Name of recording ]\n"
          f"-r | --record    [ Specify number of seconds to record to record from microphone ]\n"
          f"\n"
          f"audio-transcriber --file '~/Downloads/Federal_Reserve.mp4' --model 'large'\n"
          f"audio-transcriber --record 60 --directory '~/Downloads/' --name 'my_recording.wav' --model 'tiny'\n")


def audio_transcriber(argv):
    model = 'tiny'
    channels = 1
    rate = 44100
    file_name = 'output.wav'
    directory = os.curdir
    export_flag = False
    file = None
    seconds = 0
    language = 'en'

    try:
        opts, args = getopt.getopt(argv, "hb:c:d:ef:l:m:n:r:", ["help", "bitrate=", "channels=", "directory=", "export",
                                                                "file=", "language=", "model=", "name=", "record="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-b", "--bitrate"):
            rate = arg
        elif opt in ("-c", "--channels"):
            channels = arg
        elif opt in ("-d", "--directory"):
            directory = arg
        elif opt in ("-e", "--export"):
            export_flag = True
        elif opt in ("-f", "--file"):
            if os.path.isfile(arg):
                file = arg
            else:
                print(f"File {arg} does not exist")
                usage()
                sys.exit(2)
        elif opt in ("-l", "--language"):
            language = arg
        elif opt in ("-m", "--model"):
            if model in ['tiny', 'base', 'small', 'medium', 'large']:
                model = arg
            else:
                usage()
                sys.exit(2)
        elif opt in ("-n", "--name"):
            file_name = arg
        elif opt in ("-r", "--record"):
            seconds = arg

    if file:
        audio_transcribe = AudioTranscriber(model=model,
                                            channels=channels,
                                            rate=rate,
                                            file=file)
    else:
        audio_transcribe = AudioTranscriber(model=model,
                                            channels=channels,
                                            rate=rate,
                                            file_name=file_name,
                                            directory=directory)
        audio_transcribe.initiate_stream()
        audio_transcribe.record(seconds=seconds)
        audio_transcribe.stop_stream()
        audio_transcribe.save_stream()

    audio_transcribe.transcribe(language=language)

    if export_flag:
        audio_transcribe.export_text()


def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    audio_transcriber(sys.argv[1:])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    audio_transcriber(sys.argv[1:])
