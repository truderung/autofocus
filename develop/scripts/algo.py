from pathlib import Path
from __internals import *
import json
from addict import Dict
import numpy as np
import cv2 as cv
import time
from training_data.toolbox import *


class Stack:
    def __init__(self, img_stack, seg_height):
        self.org_img = img_stack
        self.nbr_img = len(img_stack)
        self.out_path = get_new_out_dir()
        self.segment = seg_height
        self.img_shape = False
        
    def heat_stack(self, prog):
        def __run_work(image, half_size, mask, src_path, out_path, stat):
            heater = Heater(image, half_size)
            heat_map = heater.heat(mask)
            heat_map_bar = heater.heat_bar(mask)

            stat.std[src_path.stem] = p = float(np.std(heat_map))
            stat.mean[src_path.stem] = p = float(np.mean(heat_map))
            stat.std_bar[src_path.stem] = p = float(np.std(heat_map_bar))
            stat.mean_bar[src_path.stem] = p = float(np.mean(heat_map_bar))
                
            # store heat_maps
            heater_path = Path(out_path, f"heater_{src_path.stem}").as_posix()
            stat.targets += [heater_path]
            heater_path_bar = Path(out_path, f"heater_bar_{src_path.stem}").as_posix()
            stat.targets_bar += [heater_path_bar]
            with open(heater_path, 'wb') as f:
                np.save(f, heat_map)
            with open(heater_path_bar, 'wb') as f:
                np.save(f, heat_map_bar)

            max_value = np.max(heat_map)
            stat.max_values[src_path.stem] = max_value

            max_value_bar = np.max(heat_map_bar)
            stat.max_values_bar[src_path.stem] = max_value_bar

            return max_value, max_value_bar

        # function start
        t = time.time()
        
        g = prog["gamma"][0]
        d = prog["gamma_delta"]
        gamma_range = list(range(g-d, g+d+1))
        level_distribution = prog["level_distribution"]
        width_distribution = prog["width_distribution"]
        self.max_value = np.float64(0.0)
        self.max_value_bar = np.float64(0.0)
        
        # collect debug informations
        stat = Dict()
        stat.sources = [i.as_posix() for i in self.org_img]
        stat.output = self.out_path.as_posix()
        stat.program = prog
        stat.targets = []

        # first iteration extracted because optimization (initial processes)
        img_iter = iter(self.org_img)
        src_path = next(img_iter)
        image = cv.imread(src_path.as_posix(), cv.IMREAD_GRAYSCALE)
        self.img_shape = image.shape
        half_size = (image.shape[0]//2, image.shape[1]//2)

        masker = Masker(half_size)
        self.mask = masker.get_mask(gamma_range, level_distribution, width_distribution)
        cv.imwrite(Path(self.out_path, f"mask_combined.jpg").as_posix(), self.mask)

        # create heatmaps on two diagonals separately (gamma and gamma bar)
        self.max_value, self.max_value_bar = __run_work(image, half_size, self.mask, src_path, self.out_path, stat)

        # remained iterations
        for src_path in img_iter:
            image = cv.imread(src_path.as_posix(), cv.IMREAD_GRAYSCALE)
            new_max, new_max_bar = __run_work(image, half_size, self.mask, src_path, self.out_path, stat)
            self.max_value = np.max([self.max_value, new_max])
            self.max_value_bar = np.max([self.max_value_bar, new_max_bar])

        print(" -------------------------------------------------------- ")

        stat.max_values.max = p = float(self.max_value)
        stat.max_values_bar.max = p = float(self.max_value_bar)

        stat.timings.heating = p = time.time()-t
        print(f"heating duration: {round(p, 3)}s")

        self.stat = stat

        # store json-file
        with Path(self.out_path, "params.json").open('w') as f:
            f.write(json.dumps(stat, indent=2)) 


    def eval_stack(self):
        # load heat_maps and iterate over them
        def __run_work(targets, max_value, file_ext = ""):
            hist_stack = np.empty(shape=(256,0), dtype=np.uint32)
            heat_map_stack = []

            for p in targets:
                path_stem = Path(p).stem
                with open(p, 'rb') as f:
                    heat_map = np.load(f)

                heat_map = (heat_map * 255 / max_value).astype(np.uint8)

                heat_map_stack += [(path_stem, heat_map)]
                
                # calculate the activation function
                hist = cv.calcHist(
                    images=[heat_map],
                    channels=[0],
                    mask=None,
                    histSize=[256],
                    ranges=[0, 256],
                    accumulate=False
                )
                hist_stack = np.hstack([hist_stack, hist])

            with open(Path(self.out_path, f"hist_stack{file_ext}").as_posix(), 'wb') as f:
                np.save(f, hist_stack)

            weights = exp_eval.activation(hist_stack, otsu_threshold=False)

            max_focus = 0
            h_s_stack = []
            for path_stem, heat_map in heat_map_stack:
                seg_focus = exp_eval.exp_eval_h(heat_map, weights, seg_height=self.segment)
                max_focus = np.max([max_focus, np.max(seg_focus)])
                h_s_stack += [(path_stem, heat_map, seg_focus)]
            
            return max_focus, h_s_stack

        h_s_stacks = [
            __run_work(self.stat.targets, self.stat.max_values.max),
            __run_work(self.stat.targets_bar, self.stat.max_values_bar.max, "_bar")
        ]
 
        # unite
        h_s_stack = []
        for idx, stack in enumerate(h_s_stacks[0][1]):
            path_stem, heat_map, seg_focus = stack
            seg_focus_bar = h_s_stacks[1][1][idx][2]
            seg_focus = seg_focus / h_s_stacks[0][0]
            seg_focus_bar = seg_focus_bar / h_s_stacks[1][0]
            h_s_stack += [(path_stem, heat_map, np.amax([seg_focus, seg_focus_bar], axis=0))]

        self.dft_signal = h_s_stack


    def sobel_mask(self):
        start_sobel = time.time()
        img_iter = iter(self.org_img)   

        sobel_images = []
        max_grayvalue = np.float64(0.0)
        
        for src_path in img_iter:
            image = cv.imread(src_path.as_posix(), cv.IMREAD_GRAYSCALE)

            # Compute Sobel along the vertical axis
            sobel_x = cv.Sobel(
                    src=image,
                    ddepth=cv.CV_64F,
                    dx=1,
                    dy=0,
                    ksize=3,
                    scale=1,
                    delta=0,
                    borderType=cv.BORDER_DEFAULT
                )

            # Compute Sobel along the horizontal axis
            sobel_y = cv.Sobel(
                    src=image,
                    ddepth=cv.CV_64F,
                    dx=0,
                    dy=1,
                    ksize=3,
                    scale=1,
                    delta=0,
                    borderType=cv.BORDER_DEFAULT
                )
            
            # Absolute
            abs_sobel_x = np.abs(sobel_x)
            abs_sobel_y = np.abs(sobel_y)

            # Euklid. Distance
            sobel_image = np.sqrt(
                np.multiply(abs_sobel_x, abs_sobel_x) + np.multiply(abs_sobel_y, abs_sobel_y)
                )
            
            max_grayvalue = max(max_grayvalue, np.max(sobel_image))
            sobel_images.append(sobel_image)

        sobel_images_time = time.time() - start_sobel

        # Stack-Normalisation with Max_value
        for idx, sobel_image in enumerate(sobel_images):
            sobel_images[idx] = (255*sobel_image / max_grayvalue).astype(np.uint8)

        # Otsu Threshold:
        otsu_thresholds_time = time.time()
        otsu_thresholds = []

        for thres_image in sobel_images:
            # Blur the image with GaussianBlur reduces the noise
            thres_image = cv.GaussianBlur(
                thres_image,
                (5,5),
                0
            )
            # cv.threshold binarise the "thres_image" and returns the otsu-threshold used, "ret"
            ret, _ = cv.threshold(
                thres_image,
                0,
                255,
                cv.THRESH_BINARY+cv.THRESH_OTSU
            )
            # The binary image will be ignored and only the threshold will be saved
            # Implementing a function that just computes the otsu threshold can be cheaper.
            otsu_thresholds.append(ret)
            
        # General Threshold is the the threshold that will be used for all images in the stack.
        general_thres = np.median(otsu_thresholds)

        otsu_thresholds_time = time.time() - otsu_thresholds_time
        print("Stack OTSU Threshold = ", general_thres)

        # Activation function computed using the general threshold
        weights = exp_eval.activation_Sobel(general_thres)

        # Evalute each image in the stack
        sobel_signal = []
        for sobel_image in sobel_images:
            signal = exp_eval.exp_eval_h(sobel_image, weights, self.segment)
            sobel_signal.append(
                signal
            )

        self.sobel_signal = sobel_signal
        
        print("Sobel duration: ", time.time() - start_sobel)
        start_thres_sobel = time.time()


        # Now working with binary images:

        general_thres = 2 * general_thres  # Binarisation threshold

        sobel_thre_signal = []

        for idx, thres_image in enumerate(sobel_images):
            # Binarisation
            _, thres_image = cv.threshold(
                thres_image,
                general_thres,
                255,
                cv.THRESH_BINARY
            )
            # Binary images stored for the draw and save
            sobel_images[idx] = thres_image

            # Sobel_thres_Signal calculated and saved
            sobel_thre_signal.append(
                exp_eval.sobel_eval(thres_image, self.segment)
            )

        print(
            "Sobel Thres duration: ",
            time.time() - start_thres_sobel + sobel_images_time + otsu_thresholds_time
        )

        self.sobel_thre_signal = sobel_thre_signal


        # Draw Sobel_Signal and Sobel_Thres_Signal on the binary images, then save them
        for idx, thres_image in enumerate(sobel_images): 
            draw_heat = cv.cvtColor(thres_image, cv.COLOR_GRAY2BGR)
            draw_heat = show.draw_h(
                img=draw_heat,
                seg_focus=sobel_thre_signal[idx],
                normalisation=[np.min(sobel_thre_signal), np.max(sobel_thre_signal)],
                color=(0,0,255),
                title='Sobel_Thres',
                pos=(2450,100)
            )
            draw_heat = show.draw_h(
                img=draw_heat,
                seg_focus=self.sobel_signal[idx],
                normalisation=[np.min(self.sobel_signal), np.max(self.sobel_signal)],
                color=(255,0,0),
                title='Sobel',
                pos=(2450,150)
            )
            cv.imwrite(Path(self.out_path, f"Sobel_local_{idx}.jpg").as_posix(), draw_heat)

        
        


    def laplace(self):
        start_laplace = time.time()
        img_iter = iter(self.org_img)   

        laplace_signal = []

        for src_path in img_iter:
            image = cv.imread(src_path.as_posix(), cv.IMREAD_GRAYSCALE)
            image = cv.GaussianBlur(image, (3, 3), 0)
            
            laplace_image = cv.Laplacian(
                src=image,
                ddepth=cv.CV_16S,
                ksize=3,
                scale=1,
                delta=0,
                borderType=cv.BORDER_DEFAULT
            )

            laplace_image = np.multiply(
                laplace_image,
                laplace_image
            )

            laplace_signal.append(
                exp_eval.laplace_eval(laplace_image, self.segment)
            )

        print("Laplace duration: ", time.time() - start_laplace)
        self.laplace_signal = laplace_signal



    def draw(self, DFT=True, SOBEL=True, SOBEL_THRES=False, LAPLACE=True):
        
        min_dft = 1
        for path_stem, heat_map, seg_focus in self.dft_signal:
            min_dft = min(min_dft, np.min(seg_focus))
        
        dft_signal_seg_focus = []
        idx = 0
        for path_stem, heat_map, seg_focus in self.dft_signal:
            dft_signal_seg_focus.append(seg_focus)
            if DFT:
                draw_heat = show.draw_h(
                    img=cv.cvtColor(heat_map, cv.COLOR_GRAY2BGR),
                    seg_focus=seg_focus,
                    normalisation=[min_dft, 1],
                    color=(0,255,0),
                    title='DFT',
                    pos=(2450,50)                    
                )
            else:
                draw_heat = cv.cvtColor(heat_map, cv.COLOR_GRAY2BGR)
            
            if SOBEL:
                draw_heat = show.draw_h(
                    img=draw_heat,
                    seg_focus=self.sobel_signal[idx],
                    normalisation=[np.min(self.sobel_signal), np.max(self.sobel_signal)],
                    color=(0,0,255),
                    title='Sobel',
                    pos=(2450,100)
                )

            if SOBEL_THRES:
                draw_heat = show.draw_h(
                    img=draw_heat,
                    seg_focus=self.sobel_thre_signal[idx],
                    normalisation=[np.min(self.sobel_thre_signal), np.max(self.sobel_thre_signal)],
                    color=(0,255,255),
                    title='Sobel_Thres',
                    pos=(2450,150)
                )

            if LAPLACE:
                draw_heat = show.draw_h(
                    img=draw_heat,
                    seg_focus=self.laplace_signal[idx],
                    normalisation=[np.min(self.laplace_signal), np.max(self.laplace_signal)],
                    color=(255,0,0),
                    title='Laplace',
                    pos=(2450,200)
                )
            idx +=1
            cv.imwrite(Path(self.out_path, f"draw_{path_stem}_united.jpg").as_posix(), draw_heat)
        self.dft_signal_seg_focus = dft_signal_seg_focus

    def evaluate(self):

        if self.img_shape:
            nbr_sgements = self.img_shape[0] // self.segment
        else:
            nbr_sgements = 280


        def activation_seg(nbr_segment, center, breite, min_coeff, max_coeff):
            s = (max_coeff - min_coeff) / (center - breite)
            def __sig(x):
                if x < center-breite:
                    return 0
                if x > center+breite:
                    return 0
                if x <=center:
                    return (x-breite)*s + min_coeff 
                return max_coeff - (x-center)*s
            
            return np.array([__sig(x) for x in range(0, nbr_segment)], dtype=np.float32)

        weights = activation_seg(nbr_sgements, nbr_sgements//2, nbr_sgements//4, 0.8, 1)
        dft_weights = activation_seg(140, 70, 35, 0.8, 1)

        # plt.figure()
        # plt.plot(weights)
        # plt.show()

        dft_classifier = []
        sobel_classifier = []
        sobel_thres_classifier = []
        laplace_classifier = []


        for idx_img in range(self.nbr_img):
            dft_classifier.append(
                np.sum(np.multiply(self.dft_signal_seg_focus[idx_img], dft_weights))
            )
            sobel_classifier.append(
                np.sum(np.multiply(self.sobel_signal[idx_img], weights))
            )
            sobel_thres_classifier.append(
                np.sum(np.multiply(self.sobel_thre_signal[idx_img], weights))
            )
            laplace_classifier.append(
                np.sum(np.multiply(self.laplace_signal[idx_img], weights))
            )

        dft_classifier = dft_classifier / max(dft_classifier)
        sobel_classifier = sobel_classifier / max(sobel_classifier)
        sobel_thres_classifier = sobel_thres_classifier / max(sobel_thres_classifier)
        laplace_classifier = laplace_classifier / max(laplace_classifier)

        print("Entscheidung für DFT:         Bild ", np.where(dft_classifier == 1)[0] + 1)
        print("Entscheidung für Sobel:       Bild ", np.where(sobel_classifier == 1)[0] + 1)
        print("Entscheidung für Sobel_Thres: Bild ", np.where(sobel_thres_classifier == 1)[0] + 1)
        print("Entscheidung für Laplace:     Bild ", np.where(laplace_classifier == 1)[0] + 1)

        plt.figure()
        plt.plot(dft_classifier, label='DFT')
        plt.plot(sobel_classifier, label='Sobel')
        plt.plot(sobel_thres_classifier, label='Sobel_Thres')
        plt.plot(laplace_classifier, label='Laplace')
        plt.grid(True)
        plt.legend()
        plt.show()


        
