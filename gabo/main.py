from io import BytesIO

from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
from speech_recognition import Microphone, Recognizer

m = None
for i, microphone_name in enumerate(Microphone.list_microphone_names()):
    print(microphone_name)
    if microphone_name == "default":
        m = Microphone(device_index=i)

messages = [
    {
        "role": "system",
        "content": "Tu es Gabo le robot, un robot très perfectioné qui a vécu plein d'aventure. Tu dialogues avec un enfant qui aime tes aventures"
    },
]

client = OpenAI()

r = Recognizer()
with m:
    r.adjust_for_ambient_noise(m)
    while True:
        input("Press Enter to speak...")
        audio = r.listen(m)

        print("Audio captured, sending to whisper")
        sentence = r.recognize_whisper_api(audio)
        print("Text is back:", sentence)
        messages.append({"role": "user", "content": sentence})

        print("Asking the completion")
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5
        )

        response = completion.choices[0].message.content
        print("Completion received:", response)
        messages.append({"role": "assistant", "content": response})

        print("Generating the voice")
        tts_audio = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=response,
            response_format="mp3"
        )
        with BytesIO() as buffer:
            for chunk in tts_audio.iter_bytes(1024):
                buffer.write(chunk)

            buffer.seek(0)

            audio_segment = AudioSegment.from_file(buffer, format="mp3")
            play(audio_segment)
