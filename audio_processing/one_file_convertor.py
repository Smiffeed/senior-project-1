from pydub import AudioSegment

def convert_to_wav(input_file, output_file):
    # Load the input audio file
    audio = AudioSegment.from_file(input_file)
    
    # Export as WAV
    audio.export(output_file, format="wav")

# Example usage
input_file = "test9.m4a"  # Replace with your input file path
output_file = "test9.wav"  # Replace with your desired output file path

convert_to_wav(input_file, output_file)
print(f"Conversion complete. WAV file saved as: {output_file}")

