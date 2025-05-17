import csv
import os
import subprocess

# === CONFIG ===
LABEL_ID = ["/m/0bt9lr", "/m/05tny_", "/m/07r_k2n", "/m/07qf0zm", "/m/07rc7d9", "/m/0ghcn6", "/t/dd00136", "/m/07srf8z"]
LABEL_NAME = ["Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Bay"]
INPUT_CSV = 'C:\PythonProgram\AI\FinalProject\\eval_segments.csv'  # or unbalanced_train_segments.csv
OUTPUT_DIR = [f'dog_clips/{i}' for i in LABEL_NAME]
all_dir = 'dog_clips/eval'
# === PREPARE ===
for dir in OUTPUT_DIR:
    os.makedirs(dir, exist_ok=True)

def parse_csv_and_download():
    
        

    count = 0
    with open(INPUT_CSV, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith('#'):  # skip header/comment
                continue

            label_list = row[-1].strip().replace('"', '').split(',')
            in_ornot = False
            for j in range(len(LABEL_ID)):
                if LABEL_ID[j] in label_list:
                    in_ornot = True
            if not in_ornot:
                continue

            yt_id = row[0]
            start_time = float(row[1])
            end_time = float(row[2])
            duration = end_time - start_time

            url = f"https://www.youtube.com/watch?v={yt_id}"
            output_filename = os.path.join(all_dir, f"{yt_id}_{int(start_time)}.mp4")
            file = f"{yt_id}_{int(start_time)}.mp4"
            exist = False
            for i in range(len(LABEL_ID)):
                if os.path.exists(output_filename):
                    exist = True
                    break
                if os.path.exists(LABEL_ID[i]+file):
                    exist = True
                    break
            if exist:
                print(f"[SKIP] Already downloaded: {output_filename}")
                continue
            print(f"[{count+1}] Downloading {yt_id} from {start_time}s ({duration:.1f}s)")
            cmd = [
                "yt-dlp",
                "--quiet", "--no-warnings",
                "--output", output_filename.replace(".wav", ".mp4"),
                "--download-sections", f"*{start_time}-{end_time}",
                "--recode-video", "mp4",  # ensure MP4 output
                url
            ]

            
            try:
                subprocess.run(cmd, check=True)
                count += 1
            except subprocess.CalledProcessError:
                print(f"[ERROR] Failed to download {yt_id}")
        


if __name__ == "__main__":
    parse_csv_and_download()
