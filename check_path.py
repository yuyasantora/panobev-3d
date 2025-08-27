# check_path.py
import os

# --- このパスを build_dataset.py と同じものに設定してください ---
LIDC_IDRI_ROOT = 'data/manifest-1752629384107/LIDC-IDRI/'

print(f"--- LIDC Path Diagnostic Tool ---")
print(f"Checking root path: '{LIDC_IDRI_ROOT}'")

# 絶対パスに変換して表示
abs_path = os.path.abspath(LIDC_IDRI_ROOT)

if not os.path.exists(abs_path):
    print(f"\n[❌ ERROR] The root directory does not exist: {abs_path}")
    print("Please check the LIDC_IDRI_ROOT path in the script.")
else:
    print(f"\n[✅ SUCCESS] Root directory found at: {abs_path}")
    
    patient_dirs = [d for d in os.listdir(abs_path) if d.startswith('LIDC-IDRI-')]
    
    if not patient_dirs:
        print(f"\n[❌ ERROR] No patient directories (like 'LIDC-IDRI-xxxx') found inside the root directory.")
        print("Please make sure LIDC_IDRI_ROOT points to the folder containing all patient folders.")
    else:
        print(f"\n[✅ SUCCESS] Found {len(patient_dirs)} patient directories.")
        
        first_patient = sorted(patient_dirs)[0]
        patient_path = os.path.join(abs_path, first_patient)
        print(f"\n--- Checking contents of the first patient: '{first_patient}' ---")
        
        found_dcm = False
        print("Directory structure found:")
        for root, dirs, files in os.walk(patient_path):
            level = root.replace(patient_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            
            if any(f.lower().endswith('.dcm') for f in files):
                sub_indent = ' ' * 4 * (level + 1)
                print(f"{sub_indent}--> Found .dcm files here!")
                found_dcm = True
        
        if not found_dcm:
            print("\n[❌ ERROR] No '.dcm' files were found inside this patient's directory.")
        else:
            print("\n[✅ SUCCESS] '.dcm' files were found. The path seems correct.") 