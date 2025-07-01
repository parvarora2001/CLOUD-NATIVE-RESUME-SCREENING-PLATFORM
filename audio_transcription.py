import os
import whisper

def transcribe_audio(audio_dir, output_dir):
    model = whisper.load_model("base")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(audio_dir):
        if file.endswith('.mp3') or file.endswith('.wav'):
            audio_path = os.path.join(audio_dir, file)
            result = model.transcribe(audio_path)
            out_path = os.path.join(output_dir, file.replace('.mp3', '.txt').replace('.wav', '.txt'))
            with open(out_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(result['text'])

