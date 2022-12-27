#!/usr/bin/env python
# coding: utf-8

from setuptools import setup
from audio_transcriber.version import __version__, __author__
from pathlib import Path
import os
import re
from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements


readme = Path('README.md').read_text()
version = __version__
requirements = parse_requirements(os.path.join(os.path.dirname(__file__), 'requirements.txt'), session=PipSession())
readme = re.sub(r"Version: [0-9]*\.[0-9]*\.[0-9][0-9]*", f"Version: {version}", readme)
with open("README.md", "w") as readme_file:
    readme_file.write(readme)
description = 'Transcribe your .wav .mp4 .mp3 .flac files to text or record your own audio!'

setup(
    name='audio-transcriber',
    version=f"{version}",
    description=description,
    long_description=f'{readme}',
    long_description_content_type='text/markdown',
    url='https://github.com/Knuckles-Team/subsync',
    author=__author__,
    author_email='knucklessg1@gmail.com',
    license='MIT License',
    packages=['audio_transcriber'],
    include_package_data=True,
    install_requires=[str(requirement.requirement) for requirement in requirements],
    py_modules=['audio_transcriber'],
    package_data={'audio_transcriber': ['audio_transcriber']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: Public Domain',
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={'console_scripts': ['audio-transcriber = audio_transcriber.audio_transcriber:main']},
)
