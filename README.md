# audio_preprocessor

Under development now

#### Goal
To create a single HDF file,
* that contains all the transforms of the audio files,
  * that are in a certain folder,
* where the transforms can be specified w.r.t [librosa](http://librosa.github.io)
* and so as the extensions of audio files
* with a random order for training,
* using multiprocessing.

#### How to use
1. Create  or modify [settings.json](https://github.com/keunwoochoi/audio_preprocessor/blob/master/settings.json)
2. As in the [example.py](https://github.com/keunwoochoi/audio_preprocessor/blob/master/example.py),
```	
import audio_preprocessor

nogada = audio_preprocessor.Audio_Preprocessor(settings_path='settings.json')
nogada.index()
nogada.get_permutations()
nogada.convert_all()
```

#### Credits
Test music items are from [http://www.bensound.com](http://www.bensound.com)

