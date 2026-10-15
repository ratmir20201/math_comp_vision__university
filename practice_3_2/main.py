import tarfile
import zipfile
from pathlib import Path

import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def find_dicom_dirs(root):
    found = []
    for folder in [Path(root)] + [p for p in Path(root).rglob('*') if p.is_dir()]:
        if not any(f.is_file() for f in folder.iterdir()):
            continue
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
        if ids:
            found.append((folder, ids))
    return found


print('=== Часть 1. МРТ (SimpleITK) ===')

with tarfile.open('MTR_005.tar.gz', 'r:gz') as archive:
    archive.extractall('MTR_005_data')

series_dirs = find_dicom_dirs('MTR_005_data')

if series_dirs:
    folder, series_ids = series_dirs[0]
    series_id = series_ids[0]
    print('Папка с серией:', folder)
    print('Идентификаторы серий:', series_ids)
    print('Идентификатор серии:', series_id)

    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series_id)
    print('Количество файлов для чтения:', len(file_names))
    for name in file_names[:5]:
        print(name)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()
else:
    volume_files = sorted(list(Path('MTR_005_data').rglob('*.nii.gz')) + list(Path('MTR_005_data').rglob('*.nii')) + list(Path('MTR_005_data').rglob('*.mha')))
    if not volume_files:
        raise FileNotFoundError('В архиве не найдены ни DICOM-серии, ни файлы NIfTI')
    print('DICOM-серий нет, читается файл:', volume_files[0])
    image = sitk.ReadImage(str(volume_files[0]))

print('Размер изображения (x, y, z):', image.GetSize())
print('Размеры вокселей, мм (x, y, z):', image.GetSpacing())
print('Начало координат:', image.GetOrigin())
print('Тип пикселя:', image.GetPixelIDTypeAsString())

array = sitk.GetArrayFromImage(image)
print('Форма массива numpy (z, y, x):', array.shape, '| dtype:', array.dtype)
print('Минимум, максимум, среднее:', array.min(), array.max(), round(float(array.mean()), 2))

indices = np.linspace(0, array.shape[0] - 1, 9).astype(int)
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, idx in zip(axes.ravel(), indices):
    ax.imshow(array[idx], cmap='gray')
    ax.set_title(f'Срез {idx}')
    ax.axis('off')
plt.suptitle('9 аксиальных срезов (SimpleITK)')
plt.tight_layout()
plt.show()

print('\n=== Часть 2. DICOM -> NIfTI (nibabel) ===')

with zipfile.ZipFile('1.zip') as archive:
    archive.extractall('zip_data')

nifti_files = sorted(list(Path('zip_data').rglob('*.nii.gz')) + list(Path('zip_data').rglob('*.nii')))

if nifti_files:
    nifti_path = nifti_files[0]
    print('В архиве уже есть NIfTI:', nifti_path)
else:
    zip_series = find_dicom_dirs('zip_data')
    if not zip_series:
        raise FileNotFoundError('В 1.zip не найдены ни NIfTI, ни DICOM-серии')
    zip_folder, zip_ids = zip_series[0]
    zip_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(zip_folder), zip_ids[0])
    zip_reader = sitk.ImageSeriesReader()
    zip_reader.SetFileNames(zip_names)
    zip_image = zip_reader.Execute()
    nifti_path = Path('converted.nii.gz')
    sitk.WriteImage(zip_image, str(nifti_path))
    print('DICOM-файлов:', len(zip_names), '| сохранён файл:', nifti_path)

nii = nib.load(str(nifti_path))
data = nii.get_fdata()
if data.ndim > 3:
    data = data[..., 0]
    nii = nib.Nifti1Image(data, nii.affine)

print('Форма массива numpy:', data.shape, '| dtype:', data.dtype)
print('Минимум, максимум, среднее:', round(float(data.min()), 2), round(float(data.max()), 2), round(float(data.mean()), 2))
print('Аффинная матрица:')
print(np.round(nii.affine, 3))
zooms = tuple(round(float(z), 3) for z in nii.header.get_zooms()[:3])
print('Размеры вокселей, мм:', zooms)
print('Ориентация (коды осей):', nib.aff2axcodes(nii.affine))

nx, ny, nz = data.shape
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
rows = [
    ('Сагиттальная', lambda k: data[k, :, :], np.linspace(nx * 0.25, nx * 0.75, 4).astype(int)),
    ('Корональная', lambda k: data[:, k, :], np.linspace(ny * 0.25, ny * 0.75, 4).astype(int)),
    ('Аксиальная', lambda k: data[:, :, k], np.linspace(nz * 0.25, nz * 0.75, 4).astype(int)),
]
for r, (name, getter, idxs) in enumerate(rows):
    for c, k in enumerate(idxs):
        axes[r, c].imshow(np.rot90(getter(k)), cmap='gray')
        axes[r, c].set_title(f'{name}, срез {k}')
        axes[r, c].axis('off')
plt.tight_layout()
plt.show()

new_voxel = (2.0, 2.0, 2.0)
resampled = resample_to_output(nii, voxel_sizes=new_voxel, order=1)
rdata = resampled.get_fdata()
print('Разрешение до:', data.shape, 'воксели', zooms)
print('Разрешение после:', rdata.shape, 'воксели', tuple(round(float(z), 3) for z in resampled.header.get_zooms()[:3]))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.rot90(data[:, :, nz // 2]), cmap='gray')
axes[0].set_title(f'До: {data.shape}')
axes[1].imshow(np.rot90(rdata[:, :, rdata.shape[2] // 2]), cmap='gray')
axes[1].set_title(f'После: {rdata.shape}')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

mean, std = data.mean(), data.std()
zscore = (data - mean) / std
minmax = (data - data.min()) / (data.max() - data.min())

print('Стандартизация (z-score): среднее', round(float(zscore.mean()), 4), 'std', round(float(zscore.std()), 4),
      'диапазон', round(float(zscore.min()), 2), round(float(zscore.max()), 2))
print('Нормализация (min-max): среднее', round(float(minmax.mean()), 4), 'диапазон', float(minmax.min()), float(minmax.max()))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, arr, title in zip(axes, [data, zscore, minmax], ['Исходные', 'Z-score', 'Min-max [0, 1]']):
    ax.hist(arr.ravel(), bins=100)
    ax.set_title(title)
    ax.set_yscale('log')
plt.tight_layout()
plt.show()

mid = nz // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, arr, title in zip(axes, [data, zscore, minmax], ['Исходное', 'Z-score', 'Min-max']):
    ax.imshow(np.rot90(arr[:, :, mid]), cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()