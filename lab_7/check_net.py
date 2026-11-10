import json
import urllib.request


def get(u):
    with urllib.request.urlopen(u, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


for q in ['oxford-iiit-pet', 'oxford pets', 'oxford_pet', 'pets segmentation']:
    try:
        res = get('https://huggingface.co/api/datasets?search=' + q.replace(' ', '%20') + '&limit=40')
        print('==', q, len(res))
        for d in res:
            print(d['id'], d.get('downloads'), [t for t in d.get('tags', []) if 'segm' in t or 'mask' in t])
    except Exception as e:
        print('FAIL', q, type(e).__name__, str(e)[:80])

for repo in ['Shiym/ViT-FineTune']:
    for path in ['oxford-iiit-pet', 'oxford-iiit-pet/annotations']:
        try:
            res = get(f'https://huggingface.co/api/datasets/{repo}/tree/main/{path}')
            print('==', repo, path)
            for f in res[:20]:
                print(f['type'], f['path'], f.get('size'))
        except Exception as e:
            print('FAIL', repo, path, type(e).__name__, str(e)[:80])
