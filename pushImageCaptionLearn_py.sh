#!/usr/bin/env bash

scp -r ~/source/ImageCaptionLearn_py/config/ clgrad5:~/source/ImageCaptionLearn_py/ >/dev/null
scp -r ~/source/ImageCaptionLearn_py/nn_utils/ clgrad5:~/source/ImageCaptionLearn_py/ >/dev/null
scp -r ~/source/ImageCaptionLearn_py/utils/ clgrad5:~/source/ImageCaptionLearn_py/ >/dev/null
scp ~/source/ImageCaptionLearn_py/*.py clgrad5:~/source/ImageCaptionLearn_py/ >/dev/null
scp ~/source/ImageCaptionLearn_py/*.sh clgrad5:~/source/ImageCaptionLearn_py/ >/dev/null

