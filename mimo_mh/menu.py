# ==============================================================================
# menu.py
# ==============================================================================
def display_menu():
    """
    Displays the improved model selection menu.

    Utility: Prints the interactive menu options for user selection.

    Arguments:
    - None

    Returns:
    - None
    """
    print("\n" + "="*60)
    print("MODEL SELECTION MENU")
    print("="*60)
    print("--- Individual Models ---")
    print(" 1 : Run SISO")
    print(" 2 : Run MIMO")
    print(" 3 : Run MIMO-MH")
    print("\n--- Reconciled (REC) Models Only ---")
    print(" 4 : Run SISO-REC")
    print(" 5 : Run MIMO-REC")
    print(" 6 : Run MIMO-MH-REC")
    print("\n--- Combined Models (Base + Reconciled) ---")
    print(" 7 : Run SISO (Base + REC)")
    print(" 8 : Run MIMO (Base + REC)")
    print(" 9 : Run MIMO-MH (Base + REC)")
    print(" 10 : Run ALL ELM models (Base + REC)")
    print("\n--- External Models & Groups ---")
    print(" 11 : Run Prophet + TimeGPT")
    print(" 12 : Run ALL models (ELM Base + External)")
    print(" 13 : Run ALL models (ELM REC + External)")
    print("\n--- Custom ---")
    print(" 14 : Custom model selection")
    print("="*60)

def get_user_choice(max_choice):
    """
    Gets and validates the user's menu choice.

    Utility: Prompts and validates integer input within range.

    Arguments:
    - max_choice: int, maximum valid choice.

    Returns:
    - choice: int, validated user choice.
    """
    while True:
        try:
            choice = int(input(f"Please enter your choice (1-{max_choice}): "))
            if 1 <= choice <= max_choice:
                return choice
            else:
                print(f"Invalid choice. Please enter a number between 1 and {max_choice}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_custom_model_selection():
    """
    Allows for custom selection of models and reconciliation.

    Utility: Prompts user for custom model choices, including reconciliation options for ELM models.

    Arguments:
    - None

    Returns:
    - models_to_run: list of tuples (model_name, reconcile_flag).
    """
    available_models = {
        1: "siso", 2: "mimo", 3: "mimo-mh",
        4: "prophet", 5: "timegpt"
    }
    elm_models = {"siso", "mimo", "mimo-mh"}
    
    print("\n" + "="*50)
    print("CUSTOM MODEL SELECTION")
    print("="*50)
    
    print("Available models:")
    for key, model in available_models.items():
        print(f"{key}: {model.upper()}")
        
    selected_models = []
    while True:
        try:
            choices_str = input("\nEnter the numbers of the models to run, separated by a comma (e.g., 1,3,4): ")
            choices = [int(c.strip()) for c in choices_str.split(',') if c.strip()]
            if all(c in available_models for c in choices):
                break
            else:
                print("One or more numbers are invalid. Please try again.")
        except ValueError:
            print("Invalid input. Make sure to enter numbers separated by commas.")

    models_to_run = []
    for choice in sorted(list(set(choices))): # Use sorted unique choices
        model_name = available_models[choice]
        if model_name in elm_models:
            while True:
                # Added 'b' for 'both' option
                rec_choice = input(f"  For {model_name.upper()}, run base, reconciled, or both? (n/y/b): ").lower()
                if rec_choice == 'n': # No
                    models_to_run.append((model_name, False))
                    break
                elif rec_choice == 'y': # Yes
                    models_to_run.append((model_name, True))
                    break
                elif rec_choice == 'b': # Both
                    models_to_run.append((model_name, False)) # Base version
                    models_to_run.append((model_name, True))  # Reconciled version
                    break
                else:
                    print("  Invalid response. Please enter 'n' (no), 'y' (yes), or 'b' (both).")
        else:
            models_to_run.append((model_name, False)) # Reconciliation not applicable
            
    selected_names = [f"{name.upper()}{'-REC' if rec else ''}" for name, rec in models_to_run]
    print(f"\nSelected models: {', '.join(selected_names)}")
    return models_to_run

def select_models():
    """
    Interactive function to select which models to run via the menu.

    Utility: Displays menu, gets choice, and maps to list of models/reconciliation flags.

    Arguments:
    - None

    Returns:
    - list of tuples: (model_name, reconciliation_flag).
    """
    display_menu()
    user_choice = get_user_choice(14)
    
    # Model shortcuts
    siso, siso_rec = ('siso', False), ('siso', True)
    mimo, mimo_rec = ('mimo', False), ('mimo', True)
    mimo_mh, mimo_mh_rec = ('mimo-mh', False), ('mimo-mh', True)
    prophet, timegpt = ('prophet', False), ('timegpt', False)

    # --- Individual Models ---
    if user_choice == 1: return [siso]
    elif user_choice == 2: return [mimo]
    elif user_choice == 3: return [mimo_mh]
    
    # --- Reconciled (REC) Models Only ---
    elif user_choice == 4: return [siso_rec]
    elif user_choice == 5: return [mimo_rec]
    elif user_choice == 6: return [mimo_mh_rec]
    
    # --- Combined Models (Base + Reconciled) ---
    elif user_choice == 7: return [siso, siso_rec]
    elif user_choice == 8: return [mimo, mimo_rec]
    elif user_choice == 9: return [mimo_mh, mimo_mh_rec]
    elif user_choice == 10: return [siso, siso_rec, mimo, mimo_rec, mimo_mh, mimo_mh_rec]

    # --- External Models & Groups ---
    elif user_choice == 11: return [prophet, timegpt]
    elif user_choice == 12: return [siso, mimo, mimo_mh, prophet, timegpt]
    elif user_choice == 13: return [siso_rec, mimo_rec, mimo_mh_rec, prophet, timegpt]
    
    # --- Custom ---
    elif user_choice == 14: return get_custom_model_selection()
    
    return []