from pytube import YouTube

def ladda_ner_youtube_video(url, sparväg):
   
    yt = YouTube(url)
    video = yt.streams.get_highest_resolution()
    video.download(sparväg)
    print(f"Nedladdningen är klar. Videon har sparats i: {sparväg}")

url = input("Ange URL:")  
sparväg = '../../YOUTUBE_VIDEOS_LADDANED'  #kom ihåg att skapa mapp med detta namn
ladda_ner_youtube_video(url, sparväg)
