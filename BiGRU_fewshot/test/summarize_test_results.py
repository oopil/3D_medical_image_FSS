import re
import glob
import numpy as np

def access_dice_score(path):
    fd = open(path)
    lines = fd.readlines()
    # print(len(lines),file)
    result_line = lines[-2]
    find = re.search("0.*", result_line)
    line_parts = re.split(" ", result_line)
    dice = line_parts[-2]
    return dice

def summarize(files):
    if len(files) < 5:
        # print("There is no results for this set.")
        return 0,0

    dices = []
    for file in files[:5]: #[-5:]
        dices.append(float(access_dice_score(file)))

    return dices

def print_dir_5shot(dir):
    # used original data
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()

def main():
    # bcv_dir_dice = "runs/log/bcv_dice"
    bcv_dir_dice_ce = "runs/log/bcv_dice_ce"
    ctorg_dir = "runs/log/ctorg"
    decath_dir = "runs/log/decathlon"
    # dirs = [bcv_dir_dice_ce, decath_dir]
    dirs = [bcv_dir_dice_ce, ctorg_dir, decath_dir, "runs/log/bcv_dice"]

    for dir in dirs:
        print()
        print(dir)
        for shot in [1,3,5]:
            for organ in [1,3,6,14]:
                files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
                files.sort()
                dices = summarize(files)
                avg, std = np.mean(dices), np.std(dices)
                avg = float("{:.3f}".format(avg))
                std = float("{:.4f}".format(std))
                dices = [str(dice) for dice in dices]
                dice_str = ",".join(dices)
                # print(dir, organ, shot)
                if avg*std!=0:
                    print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")

    dir = "runs/log/bcv_dice_ce_bladder"
    print(dir)
    shot=5
    organ=14
    files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
    files.sort()
    for i, file in enumerate(files):
        dice = float(access_dice_score(file))
        print(i, dice)
        # dices = summarize(files)
        # print(dices)
        # avg, std = np.mean(dices), np.std(dices)
        # avg = float("{:.3f}".format(avg))
        # std = float("{:.4f}".format(std))
        # dices = [str(dice) for dice in dices]
        # dice_str = ",".join(dices)
        # # print(dir, organ, shot)
        # if avg * std != 0:
        #             print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()

    assert False
    dir = "runs/log/bcv_super"
    print(dir)
    for shot in [1, 3, 5]:
        for organ in [1, 3, 6, 14]:
            files = glob.glob(f"{dir}/*_super{organ}_*{shot}shot*")
            files.sort()
            dices = summarize(files)
            avg, std = np.mean(dices), np.std(dices)
            avg = float("{:.3f}".format(avg))
            std = float("{:.4f}".format(std))
            dices = [str(dice) for dice in dices]
            dice_str = ",".join(dices)
            # print(dir, organ, shot)
            if avg * std != 0:
                print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()

    dir = "runs/log/bcv_dice_ce_7slice"
    print(dir)
    shot=1
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()

    dir = "runs/log/bcv_dice_ce_9slice"
    print(dir)
    shot=1
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()

    dir = "runs/log/bcv_dice_ce_11slice"
    print(dir)
    shot=1
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()

    dir = "runs/log/bcv_dice_ce_5shot_7slice"
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()


    # bcv_finetuning_dir = "runs/log/bcv_finetuning_supp0"
    # print(bcv_finetuning_dir)
    # for organ in [1, 3, 6, 14]:
    #     for update in [0, 5, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200, 250, 300]:
    #         files = glob.glob(f"{bcv_finetuning_dir}/*_{organ}_*{update}update*")
    #         files.sort()
    #         dices = access_dice_score(files[0])
    #         print(f"organ:{organ}, {shot}shot, {update}update,{dices}")
    # print()

    # bcv_finetuning_dir = "runs/log/bcv_finetuning_supp0_v2"
    # print(bcv_finetuning_dir)
    # for organ in [1, 3, 6, 14]:
    #     for update in [0, 200, 400, 600, 1000]:
    #         files = glob.glob(f"{bcv_finetuning_dir}/*_{organ}_*{update}update*")
    #         files.sort()
    #         dices = access_dice_score(files[0])
    #         print(f"organ:{organ}, {shot}shot, {update}update,{dices}")
    # print()

    # ctorg_finetuning_dir = "runs/log/ctorg_finetuning_supp30_b2"
    # print(ctorg_finetuning_dir)
    # for organ in [3, 6, 14]:
    #     for update in [0, 30, 60, 90, 120, 150]:
    #         files = glob.glob(f"{ctorg_finetuning_dir}/*_{organ}_*{update}update*")
    #         files.sort()
    #         dices = access_dice_score(files[0])
    #         print(f"organ:{organ}, {shot}shot, {update}update,{dices}")
    # print()


    decathlon_finetuning_dir = "runs/log/decathlon_finetuning_supp0"
    print(decathlon_finetuning_dir)
    for organ in [1, 6]:
        for update in [0, 10, 20, 30, 40, 50]:
            files = glob.glob(f"{decathlon_finetuning_dir}/*_{organ}_*{update}update*")
            files.sort()
            dices = access_dice_score(files[0])
            print(f"organ:{organ}, {shot}shot, {update}update,{dices}")
    print()

    dir = "runs/log/bcv_100update"
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")

    dir = "runs/log/decathlon_100update"
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")


    dir = "runs/log/ctorg_0update_v1_fast"
    # used original data
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")

    dir = "runs/log/ctorg_10update_v1_fast"
    # used original data
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()

    dir = "runs/log/ctorg_20update_v1_fast"
    # used original data
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()

    dir = "runs/log/ctorg_10update_v1_fast_ce"
    # used original data
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    print()


    dir = "runs/log/ctorg_50update_v1_fast"
    # used original data
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")

    dir = "runs/log/ctorg_50update_v1"
    # used original data
    print(dir)
    shot=5
    for organ in [1, 3, 6, 14]:
        files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
        files.sort()
        dices = summarize(files)
        avg, std = np.mean(dices), np.std(dices)
        avg = float("{:.3f}".format(avg))
        std = float("{:.4f}".format(std))
        dices = [str(dice) for dice in dices]
        dice_str = ",".join(dices)
        # print(dir, organ, shot)
        if avg * std != 0:
            print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")


    # print_dir_5shot("runs/log/ctorg_0update_v1")
    # print_dir_5shot("runs/log/ctorg_5update_v1")
    # print_dir_5shot("runs/log/ctorg_10update_v1")
    # print_dir_5shot("runs/log/ctorg_15update_v1")
    # print_dir_5shot("runs/log/ctorg_20update_v1")
    # print_dir_5shot("runs/log/ctorg_40update_v1")

    # print_dir_5shot("runs/log/ctorg_0update_v2_fast")
    # print_dir_5shot("runs/log/ctorg_10update_v2_fast")
    # print_dir_5shot("runs/log/ctorg_40update_v2_fast")

    # dir = "runs/log/ctorg_finetuning_supp0_fast_v3_2"
    # print(dir)
    # for organ in [3, 6, 14]:
    #     for update in [0, 5, 10,20,40,60,100,200]:
    #         files = glob.glob(f"{dir}/*_{organ}_*{update}update*")
    #         files.sort()
    #         dices = access_dice_score(files[0])
    #         print(f"organ:{organ}, {shot}shot, {update}update,{dices}")
    # print()

    # dir = "runs/log/ctorg_kidney"
    # organ=3
    # print(dir)
    # for shot in [3, 5]:
    #     files = glob.glob(f"{dir}/*_{organ}_*{shot}shot*")
    #     files.sort()
    #
    #     dices = []
    #     for file in files[5:5+5]:
    #         dices.append(float(access_dice_score(file)))
    #
    #     avg, std = np.mean(dices), np.std(dices)
    #     avg = float("{:.3f}".format(avg))
    #     std = float("{:.4f}".format(std))
    #     dices = [str(dice) for dice in dices]
    #     dice_str = ",".join(dices)
    #
    #     if avg * std != 0:
    #         print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")
    # print()


if __name__=="__main__":
    main()
