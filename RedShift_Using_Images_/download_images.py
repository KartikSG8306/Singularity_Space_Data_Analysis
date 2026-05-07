import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(row, index):
    ra = row['ra']
    dec = row['dec']
    redshift = row['redshift']
    width = 128
    height = 128
    scale = 0.2
    
    url = f"https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&width={width}&height={height}&scale={scale}"
    image_filename = f"galaxy_{index}.jpg"
    image_path = os.path.join("images", image_filename)
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return {'image': image_filename, 'redshift': redshift}
        else:
            return None
    except Exception:
        return None

def main():
    print("Loading dataset...")
    df = pd.read_csv("SDSS_DR18.csv")
    
    galaxies = df[df['class'] == 'GALAXY']
    print(f"Found {len(galaxies)} galaxies in the dataset.")
    
    num_samples = min(2000, len(galaxies))
    sampled_galaxies = galaxies.sample(n=num_samples, random_state=42).reset_index(drop=True)
    
    os.makedirs('images', exist_ok=True)
    
    labels_data = []
    
    print("Downloading images using multithreading...")
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(download_image, row, i) for i, row in sampled_galaxies.iterrows()]
        
        for future in tqdm(as_completed(futures), total=num_samples):
            result = future.result()
            if result is not None:
                labels_data.append(result)
                
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv("labels.csv", index=False)
    print(f"Finished! Successfully downloaded {len(labels_df)} images.")
    print("Saved label mappings to labels.csv.")

if __name__ == "__main__":
    main()
