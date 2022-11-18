import tile_processing as tp
import os
import sys

root = os.path.join('../', 'data', 'MoNuSeg_WSI')
SLIDE_SIZE = 512
OUT_ROOT = '../monuseg_tiles_%dx%d/' % (SLIDE_SIZE, SLIDE_SIZE)


def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()


def run(n_imgs=None):
    filenames = [filename for filename in os.listdir(root) if filename.endswith('.svs')]

    files = []
    for filename in filenames:
        files.append(os.path.join(root, filename))

    n_imgs = len(files)

    for _id in range(n_imgs):
        s_name = files[_id]

        # Get the name of the tile file.
        out_name = os.path.basename(s_name)

        # Get the name of tile without the extension.
        out_name = os.path.splitext(out_name)[0]

        savepath = os.path.join(OUT_ROOT, out_name)

        if os.path.exists(savepath):
            print('%s already exists. No need to extract tiles again.' % (savepath))
            continue
        else:
            print('%s does not exist. Proceeding with extraction.' % (savepath))

        tp.slide2tiles(s_name, OUT_ROOT, SLIDE_SIZE, savepath)

        write_flush('Finished extracting tiles for %3d/%3d patients.' % (_id, n_imgs))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        n_imgs = int(sys.argv[1])
    else:
        n_imgs = None

    print('Tile size = %d. Writing results to %s.' % (SLIDE_SIZE, OUT_ROOT))

    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)

    run(n_imgs=n_imgs)
