import whisper
import os
import json
import torchaudio
import argparse
import torch
from config import config

lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
        }
def transcribe_one(audio_path):
    try:
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        lang = max(probs, key=probs.get)
        # decode the audio
        options = whisper.DecodingOptions(beam_size=5)
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)
        return lang, result.text
    except Exception as e:
        print(e)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="J")
    parser.add_argument("--whisper_size", default="medium")
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "ZH|",
        }
    elif args.languages == "J":
        lang2token = {
            'ja': "JP|",
        }
    assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
    model = whisper.load_model(args.whisper_size)
    parent_dir = config.resample_config.in_dir
    wavs_dir = config.resample_config.out_dir
    speaker_names = list(os.walk(parent_dir))[0][1]
    speaker_annos = []
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    # resample audios
    # 2023/4/21: Get the target sampling rate
    with open(config.train_ms_config.config_path, 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    processed_files = 0
    os.makedirs(wavs_dir, exist_ok=True)
# 处理每个说话人
    for speaker in speaker_names:
        input_dir = os.path.join(parent_dir, speaker)
        output_dir = os.path.join(wavs_dir, speaker)
        os.makedirs(output_dir, exist_ok=True)

        for i, wavfile in enumerate(os.listdir(input_dir)):
            if wavfile.startswith("processed_"):
                continue

            try:
                # 加载音频
                input_path = os.path.join(input_dir, wavfile)
                wav, sr = torchaudio.load(input_path, normalize=True, channels_first=True)
                
                # 预处理（单声道/重采样/长度检查）
                wav = wav.mean(dim=0).unsqueeze(0)
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
                if wav.shape[1] / target_sr > 20:
                    print(f"{wavfile} too long, skipping")
                    continue
                
                # 保存处理后的音频
                output_path = os.path.join(output_dir, f"processed_{i}.wav")
                torchaudio.save(output_path, wav, target_sr)
                
                # 转录文本
                lang, text = transcribe_one(output_path)
                if lang not in lang2token:
                    print(f"{lang} not supported, skipping")
                    continue
                    
                # 保存标注
                speaker_annos.append(f"{output_path}|{speaker}|{lang2token[lang]}{text}\n")
                processed_files += 1
                print(f"Progress: {processed_files}/{total_files}")
                
            except Exception as e:
                print(f"Error processing {wavfile}: {str(e)}")
                continue

    # # clean annotation
    # import argparse
    # import text
    # from utils import load_filepaths_and_text
    # for i, line in enumerate(speaker_annos):
    #     path, sid, txt = line.split("|")
    #     cleaned_text = text._clean_text(txt, ["cjke_cleaners2"])
    #     cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
    #     speaker_annos[i] = path + "|" + sid + "|" + cleaned_text
    # write into annotation
    if len(speaker_annos) == 0:
        print("Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links.")
        print("this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
    with open(config.preprocess_text_config.transcription_path, 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)

    # import json
    # # generate new config
    # with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
    #     hps = json.load(f)
    # # modify n_speakers
    # hps['data']["n_speakers"] = 1000 + len(speaker2id)
    # # add speaker names
    # for speaker in speaker_names:
    #     hps['speakers'][speaker] = speaker2id[speaker]
    # # save modified config
    # with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
    #     json.dump(hps, f, indent=2)
    # print("finished")
