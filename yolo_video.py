import sys
import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image

img_end = ['png', 'jpg', 'jpeg']
def detect_img(yolo, testdir):
    #while True:
    
    for img in os.listdir(testdir):
        for ending in img_end:
            if not img.endswith(ending):
                continue
            #img = input('Input image filename:')
            image = Image.open(img)
            r_image = yolo.detect_image(image)
            if r_image.mode in ("RGBA", "P"): r_image = r_image.convert("RGB")
            os.makedirs(os.path.join(testdir,'boundedImages'), exist_ok=True)
            print('Saving: {}'.format(os.path.join(testdir,'boundedImages') + '/' + img.split("/")[-1].split('.')[0] + '_model_out.jpg'))
            r_image.save(os.path.join(testdir,'boundedImages') + '/' + img.split("/")[-1].split('.')[0] + '_model_out.jpg')
                #try:
                #    image = Image.open(img)
                #    r_image = yolo.detect_image(image)
                #    r_image.show()
                #except:
                #    print('Open Error! Try again!')
                #    continue
                #else:
                #    r_image = yolo.detect_image(image)
                #    r_image.show()
            yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, dest='model_path',
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str, dest='anchors_path',
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, dest='classes_path',
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    
    parser.add_argument(
        "--testdir", nargs='?', type=str,required=False,default=os.getcwd(),
        help = "input path for testimages"
    )


    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        
        detect_img(YOLO(**vars(FLAGS)), FLAGS.testdir)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
