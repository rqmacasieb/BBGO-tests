import subprocess
import shutil
import os

def run_model():

    shutil.copy(os.path.join("pest", "pump.dat"),
                os.path.join("model", "preproc", "wel-decvar", "wel", "pump.dat"))

    os.chdir(os.path.join("model", "preproc", "wel-decvar", "wel"))
    subprocess.run(["./pumpdist"], check=True)
    
    # print(os.getcwd())
    os.chdir("../../../")
    shutil.copy(os.path.join("preproc", "wel-decvar", "wel", "prams36_opt.wel"), 
                os.path.join("input", "prams36_opt.wel"))

    subprocess.run(["./vfm-mf2k", "prams36.nam"], check=True)
    
    # Run with input redirection
    os.chdir("postproc/")
    with open("zonebudget_inputs.txt", 'r') as f:
        subprocess.run(
            ["./zonebudget"],
            stdin=f,
            check=True)
    os.chdir("..")

    input_file = "postproc/swizones.lst"
    output_csv = "postproc/zonebudget_by_zone.csv"
    start_sp = 121
    end_sp = 252
    zones_to_sum = [z for z in range(1, 28) if z not in [23, 27]]

    import model.postproc.swi_postproc as swi_postproc
    swi_postproc.extract_zonebudget_data(input_file, output_csv, budget_term="CONSTANT HEAD", section="IN", start_sp=start_sp, end_sp=end_sp)
    swi_postproc.analyze_zones(output_csv, zones_to_sum, start_stress_period=start_sp)    

    import model.postproc.obj_func as obj_func
    obj_func.calc_obj()

    os.chdir("../")

    shutil.copy(os.path.join("model", "preproc", "wel-decvar", "wel", "pump_tds.txt"),
                os.path.join("pest", "pump_tds.txt"))

if __name__ == "__main__":
    run_model()
