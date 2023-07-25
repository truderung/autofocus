import time

t0_init = time.time()

# from algo import Stack
import sys
import os
import erlastic

import elixir_utils as eu
from autofocus.toolbox.utils import *

PIX_TO_MM = 0.00274  # in mm
SEG_HEIGHT = 10  # in pixel

# "Unique" id
pid = os.getpid()
host = os.uname().nodename
kernel = os.uname().release

t = eu.T(t0_init)

def transform(bin_img):
    # transforms bin_img to np.array
    # Not done yet
    pass


def main():
    # Re-route stdout to stderr so that PythonPort works as expected
    # while python code may print without specifying file=sys.stderr
    std_org = sys.stdout
    sys.stdout = sys.stderr

    eu.log(f"### {pid}: Opening erlastic port")
    mailbox, port = erlastic.port_connection()

    (processing_step, map) = eu.convert_elixir_to_python(next(mailbox))

    def __send_safe(step: str, atom_string: str, *values):
        '''
        @brief Send response messages safely to the Elixir process via the mailbox.
            
            Step:
                0 : y_correction        => value = correction in mm
                1 : z_start_position    => value = image index im Stack
                2 : z_adjust            => value = correction in mm
                3 : new_images_top      => value = Number of new images
                4 : new_images_bottom   => value = Number of new images

        @param step: str        The processing step being responded to.
        @param atom_string: str The type of response message (e.g., "ok", "error").
        @param *values: tuple   Values to be sent as part of the response message.

        @return: bool: True if the response was sent successfully, False otherwise.
        '''

        try:
            port.send((erlastic.Atom(atom_string),) + tuple((step, values)))
            return True
        except StopIteration:
            eu.log(f"### {pid}: Worker terminated")
        except Exception as e:
            eu.log(f"### {pid}: Data transmission in step {step} failed. Error type {str(type(e))} \n\r{e}")
        return False

    # Initializations and Setup

    size = (5320, 2800)
    half_size = (size[0] // 2, size[1] // 2)

    gamma_delta = 7
    level_distribution = [0.005, 0.01, 0.02, 0.05, 0.1]
    width_distribution = [30, 60, 100]

    gamma = round(map["braiding_angle"])
    roi = map["roi"]

    gamma_range = list(range(gamma - gamma_delta, gamma + gamma_delta + 1))

    # Create the DFT mask
    try:
        mask = create_DFT_mask(
            half_size, gamma_range, level_distribution, width_distribution
        )

    except Exception as e:
        eu.log(f"### {pid}: Mask could not be created")
        eu.log(f"### {pid}: Received an unhandled exception in step of type {type(e), e}")
        __send_safe(str(processing_step), "error", str(e))
        exit(0)


    while True:
        try:
            (processing_step, map) = eu.convert_elixir_to_python(
                next(mailbox)
            )
        except StopIteration:
            eu.log(f"### {pid}: Worker terminated")
            break
        except Exception as e:
            eu.log(f"### {pid}: Worker terminated, reason: {type(e), e}")
            break
        except:
            eu.log(f"### {pid}: Worker terminated, reason: unknown")
            break

        ######################################################################################
        ######################################################################################
                # Step 0: Calibrate
        ######################################################################################
        ######################################################################################
        
        if processing_step == 0: # Calibrate
            try:
                images = transform(map["data"])
            except Exception as e:
                eu.log(f"### {pid}: Binary Images could not be converted")
                eu.log(
                    f"### {pid}: Received an unhandled exception in step of type {type(e), e}"
                )
                __send_safe(str(processing_step), "error", str(e))
                exit(0)

            heat_maps = []
            heat_maps_bar = []

            if (
                max_value not in locals()
                or max_value not in globals()
                or max_value_bar not in locals()
                or max_value_bar not in globals()
            ):
                max_value = 0
                max_value_bar = 0

            for image in images:
                res, res_bar = heat_double(image, mask)
                max_value = max(max_value, np.max(res))
                max_value_bar = max(max_value_bar, np.max(res_bar))
                heat_maps.append(res)
                heat_maps_bar.append(res_bar)

            h_segments_stack = eval_stack(
                heat_maps, max_value, heat_maps_bar, max_value_bar, SEG_HEIGHT
            )
            y_flag, y_correction, units = calibrate_y(h_segments_stack)

            if y_flag == False:
                LOOP = True

                while LOOP:
                    __send_safe(str(3), "ok", str(units)) # processing_step 3 = new_images_top

                    try:
                        (processing_step, map) = eu.convert_elixir_to_python(
                            next(mailbox)
                        )
                    except StopIteration:
                        eu.log(f"### {pid}: Worker terminated while Y_Calibration")
                        exit(0)
                    except Exception as e:
                        eu.log(f"### {pid}: Worker terminated while Y_Calibration, reason: {type(e), e}")
                        exit(0)
                    except:
                        eu.log(f"### {pid}: Worker terminated while Y_Calibration, reason: unknown")
                        exit(0)
                        
                    if processing_step != 3: # processing_step 3 = new_images_top
                        eu.log(
                            f"### {pid}: Worker terminated while Y_Calibration, reason: unknown processing step"
                        )
                        exit(0)

                    try:
                        images = transform(map["data"])
                    except Exception as e:
                        eu.log(f"### {pid}: Binary Images could not be converted")
                        eu.log(
                            f"### {pid}: Received an unhandled exception in step of type {type(e), e}"
                        )
                        __send_safe(str(processing_step), "error", str(e))
                        exit(0)

                    heat_maps = []
                    heat_maps_bar = []

                    for image in images:
                        res, res_bar = heat_double(image, mask)
                        heat_maps.append(res)
                        heat_maps_bar.append(res_bar)

                    new_h_segments_stack = eval_stack(
                        heat_maps, max_value, heat_maps_bar, max_value_bar, SEG_HEIGHT
                    )
                    h_segments_stack = np.hstack((new_h_segments_stack, h_segments_stack))
                    y_flag, y_correction, units = calibrate_y(h_segments_stack)

                    if y_flag == True:
                        LOOP = False

            if y_correction != 0: ## Weg
                dist = y_correction * PIX_TO_MM * SEG_HEIGHT
                __send_safe(str(0), "ok", str(dist)) # y_correction = 0

            # Case were the stack was taken far above the Stent: Solution can be to re-run of the y_calib func. to update units:
            LOOP = True
            while LOOP:
                if (len(h_segments_stack) - units) < 10:
                    nbr_new_images = 10 - (len(h_segments_stack) - units)

                    __send_safe(str(4), "ok", str(nbr_new_images)) # new_images_bottom = 4

                    try:
                        (processing_step, map) = eu.convert_elixir_to_python(
                            next(mailbox)
                        )
                    except StopIteration:
                        eu.log(f"### {pid}: Worker terminated while Z_Calibration")
                        exit(0)
                    except Exception as e:
                        eu.log(f"### {pid}: Worker terminated while Z_Calibration, reason: {type(e), e}")
                        exit(0)
                    except:
                        eu.log(f"### {pid}: Worker terminated while Z_Calibration, reason: unknown")
                        exit(0)

                    if processing_step != 4: # processing_step 4 = new_images_bottom
                        eu.log(
                            f"### {pid}: Worker terminated while Z_Calibration, reason: unknown processing step"
                        )
                        exit(0)

                    try:
                        images = transform(map["data"])
                    except Exception as e:
                        eu.log(f"### {pid}: Binary Images could not be converted")
                        eu.log(
                            f"### {pid}: Received an unhandled exception in step of type {type(e), e}"
                        )
                        __send_safe(str(processing_step), "error", str(e))
                        exit(0)

                    heat_maps = []
                    heat_maps_bar = []

                    for image in images:
                        res, res_bar = heat_double(image, mask)
                        heat_maps.append(res)
                        heat_maps_bar.append(res_bar)

                    new_h_segments_stack = eval_stack(
                        heat_maps, max_value, heat_maps_bar, max_value_bar, SEG_HEIGHT
                    )
                    h_segments_stack = np.hstack((h_segments_stack, new_h_segments_stack))

                    _, y_correction_new, units = calibrate_y(h_segments_stack)

                else:
                    LOOP = False

            if (y_correction_new in locals() or y_correction_new in globals()) and (y_correction_new != y_correction):
                dist = (y_correction_new - y_correction) * PIX_TO_MM * SEG_HEIGHT
                __send_safe(str(0), "ok", str(dist)) # y_correction = 0
                y_correction = y_correction_new

            best = evaluate_roi(roi, y_correction, h_segments_stack, SEG_HEIGHT)
            start_h_segment = h_segments_stack[best]
            __send_safe(str(1), "ok", str(best))



        ######################################################################################
        ######################################################################################
                # Step 1: Adjust
        ######################################################################################
        ######################################################################################



        elif processing_step == 1: # processing_step 1 = adjust
            try:
                images = transform(map["data"])
            except Exception as e:
                eu.log(f"### {pid}: Binary Images could not be converted")
                eu.log(
                    f"### {pid}: Received an unhandled exception in step of type {type(e), e}"
                )
                __send_safe(str(processing_step), "error", str(e))
                exit(0)

            heat_maps = []
            heat_maps_bar = []

            if (
                max_value not in locals()
                or max_value not in globals()
                or max_value_bar not in locals()
                or max_value_bar not in globals()
            ):
                max_value = 0
                max_value_bar = 0

            for image in images:
                res, res_bar = heat_double(image, mask)
                max_value = max(max_value, np.max(res))
                max_value_bar = max(max_value_bar, np.max(res_bar))
                heat_maps.append(res)
                heat_maps_bar.append(res_bar)

            h_segments_stack = eval_stack(
                heat_maps, max_value, heat_maps_bar, max_value_bar, SEG_HEIGHT
            )
            best = evaluate_roi(roi, 0, h_segments_stack, SEG_HEIGHT)

            if best == 0 or best == (len(h_segments_stack) - 1):
                LOOP = True
                nbr_new_images = 5
                while LOOP:
                    if best == 0:
                        __send_safe(str(3), "ok", str(nbr_new_images)) # 3 = new_images_top
                    else:
                        __send_safe(str(4), "ok", str(nbr_new_images)) # 4 = new_images_bottom


                    try:
                        (processing_step, map) = eu.convert_elixir_to_python(
                            next(mailbox)
                        )
                    except StopIteration:
                        eu.log(f"### {pid}: Worker terminated while Adjusting")
                        exit(0)
                    except Exception as e:
                        eu.log(f"### {pid}: Worker terminated while Adjusting, reason: {type(e), e}")
                        exit(0)
                    except:
                        eu.log(f"### {pid}: Worker terminated while Adjusting, reason: unknown")
                        exit(0)

                    if (processing_step != 4 and best == (len(h_segments_stack) - 1)
                    ) or (processing_step != 3 and best == 0): # processing_step 3 = new_images_top, processing_step 4 = new_images_bottom
                        eu.log(f"### {pid}: Worker terminated while Adjusting, reason: unknown processing step")
                        exit(0)

                    try:
                        images = transform(map["data"])
                    except Exception as e:
                        eu.log(f"### {pid}: Binary Images could not be converted")
                        eu.log(f"### {pid}: Received an unhandled exception in step of type {type(e), e}")
                        __send_safe(str(processing_step), "error", str(e))
                        exit(0)

                    heat_maps = []
                    heat_maps_bar = []

                    for image in images:
                        res, res_bar = heat_double(image, mask)
                        heat_maps.append(res)
                        heat_maps_bar.append(res_bar)

                    new_h_segments_stack = eval_stack(
                        heat_maps, max_value, heat_maps_bar, max_value_bar, SEG_HEIGHT
                    )

                    if best == 0:
                        h_segments_stack = np.hstack(
                            (new_h_segments_stack, h_segments_stack)
                        )
                    else:
                        h_segments_stack = np.hstack(
                            (h_segments_stack, new_h_segments_stack)
                        )

                    best = evaluate_roi(roi, 0, h_segments_stack, SEG_HEIGHT)

                    if best != 0 and best != (len(h_segments_stack) - 1):
                        LOOP = False

            start_h_segment = h_segments_stack[best]
            __send_safe(str(1), "ok", str(best))


        ######################################################################################
        ######################################################################################
                # Step 2: Predict
        ######################################################################################
        ######################################################################################

        elif processing_step == 2: # processing_step 2 = predict
            if (
                max_value not in locals()
                or max_value not in globals()
                or max_value_bar not in locals()
                or max_value_bar not in globals()
                or start_h_segment not in locals()
                or start_h_segment not in globals()
            ):
                eu.log(f"### {pid}: Prediction can not be started.")
                eu.log(f"### {pid}: System is not Calibrated and not adjusted")
                exit(0)

            image = map["data"]
            region = map["region"]
            region = region // SEG_HEIGHT
            res, res_bar = heat_double(image, mask)
            max_value = max(max_value, max(res))
            max_value_bar = max(max_value_bar, max(res_bar))
            new_h_segments = eval_stack(res, max_value, res_bar, max_value_bar, SEG_HEIGHT)

            shift = predict(start_h_segment, new_h_segments, region)

            __send_safe(str(2), "ok", str(shift)) 


        
        else:
            eu.log(f"### {pid}: Received an unknown processing step")
            __send_safe(processing_step, "error", str(e))
            exit(0)

