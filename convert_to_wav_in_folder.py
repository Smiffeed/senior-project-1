import os
from pydub import AudioSegment
from tqdm import tqdm

def convert_to_wav(input_file, output_file):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        return True
    except Exception as e:
        print(f"Error converting {input_file} to WAV: {str(e)}")
        return False

def convert_folder_to_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path):
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_folder, output_filename)
            if convert_to_wav(input_path, output_path):
                print(f"Converted: {filename} -> {output_filename}")
            else:
                print(f"Failed to convert: {filename}")

if __name__ == "__main__":
    input_folder = "./Audio/Non-Profanity"  # Change this to your input folder path
    output_folder = "./Converted_WAV/Non-profanity"  # Change this to your desired output folder path
    
    print(f"Converting audio files in {input_folder} to WAV format...")
    convert_folder_to_wav(input_folder, output_folder)
    print("Conversion complete.")
