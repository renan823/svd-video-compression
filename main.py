from src.videobw import VideoBW

def main():
    video = VideoBW()
    video.read("stop-motion.mp4")
    video.write()


if __name__ == "__main__":
    main()
