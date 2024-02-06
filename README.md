## SupRebuild

SupRebuild is a tool to fix SUP subtitle files, so that they can work properly with the Android app designed to use [media/exoplayer](https://github.com/androidx/media "media/exoplayer"). (ex: [Just (Video) Player](https://github.com/moneytoo/Player "Just (Video) Player"), [jellyfin-androidtv](https://github.com/jellyfin/jellyfin-androidtv "jellyfin-androidtv"), [findroid](https://github.com/jarnedemeulemeester/findroid "findroid").)

Main fixes:

- Some subtitles can cause part of the color to become transparent. (ex: sample/Conan24.cut.mkv)
- Multiple subtitle objects cannot be displayed correctly. (ex: sample/Conan24.cut.mkv)
- Continuous subtitles flickering problem. (ex: sample/Continue.Test.mkv)

## Usage

Process a input ".sup" file and generate a output ".mod.sup" file.

`python3 supRebuild.py [Parameters] supFile`

### Parameters
```
-d, --debug                    display debug message 
-xml                           output XML/PNG files to folder "./xmlPng"
-fps {23.976, 24, 25, 29.97}   frame rate for XML file, default = 25
```
