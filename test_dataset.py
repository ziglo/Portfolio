from dataset import DogMoodDataset

# Use the absolute path to your data directory
dataset = DogMoodDataset(image_dir=r'c:\Users\Austin\Desktop\Projects\dog-mood-detector\src\data')
print(f"Total images: {len(dataset)}")
img, label = dataset[0]
print(f"First label: {label}, Image size: {img.size}")