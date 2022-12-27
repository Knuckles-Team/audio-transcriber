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


class AudioTranscriber:
    def __init__(self, model='base', channels=1, rate=44100, file_name='output.wav', directory=os.curdir, file=None):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = channels
        self.rate = rate
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.file_name = file_name
        self.directory = directory
        if file:
            self.file = file
        else:
            os.path.join(self.directory, self.file_name)
        self.stop = False
        self.initiate_stream()
        self.model = whisper.load_model(model)

    def initiate_stream(self):
        self.stream = self.pyaudio_instance.open(format=self.format,
                                                 channels=self.channels,
                                                 rate=self.rate,
                                                 input=True,
                                                 frames_per_buffer=self.chunk)

    def set_file(self, file):
        self.file = file
        self.file_name = os.path.basename(file)
        self.directory = os.path.dirname(file)

    def set_file_name(self, file_name):
        self.file_name = file_name
        self.set_file(os.path.join(self.directory, self.file_name))

    def set_directory(self, directory):
        self.directory = directory
        self.set_file(os.path.join(self.directory, self.file_name))

    def set_rate(self, rate):
        self.rate = rate

    def set_channels(self, channels):
        self.channels = channels

    def set_model(self, model):
        if model in ['tiny', 'base', 'small', 'medium', 'large']:
            self.model = whisper.load_model(model)
        else:
            print('Model does not exist, please choose from: tiny, base, small, medium, or large')

    def record(self, seconds=0):
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

    def transcribe(self):
        start_time = datetime.datetime.now()
        print(f"Started: {start_time}")
        output = self.model.transcribe(self.file)
        end_time = datetime.datetime.now()
        print(f"Ended: {end_time}\nTime Elapsed: {end_time - start_time}")
        print(f"Output: \n{output}")
        for segment in output['segments']:
            second = int(segment['start'])
            second = second - (second % 5)
            print(f'Second: {second} - Segment: \n{segment}\n\n')


def usage():
    print(f"Usage: \n"
          f"-h | --help      [ See usage for script ]\n"
          f"-b | --bitrate   [ Bitrate to use during recording ]\n"
          f"-c | --channels  [ Number of channels to use during recording ]\n"
          f"-d | --directory [ Directory to save recording ]\n"
          f"-f | --file      [ File to transcribe ]\n"
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
    file = None
    seconds = 0
    try:
        opts, args = getopt.getopt(argv, "hb:c:d:f:m:n:r:", ["help", "bitrate=", "channels=", "directory=", "file=",
                                                             "model=", "name=", "record="])
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
        elif opt in ("-f", "--file"):
            file = arg
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
        audio_transcribe.record(seconds=seconds)
        audio_transcribe.stop_stream()
        audio_transcribe.save_stream()

    audio_transcribe.transcribe()


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
