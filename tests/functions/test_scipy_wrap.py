
from c2i2o.functions.scipy_wrap import create_pydantic_models_for_scipy_stats



def test_scipy_wrap():
    """ Test the wrapper that creates pydantic models for scipy classes"""
    
    models = create_pydantic_models_for_scipy_stats()
    
    # --- Demonstration ---
    print("\n--- Generated Models Summary (First 5) ---")
    for i, (dist_name, model) in enumerate(models.items()):
        if i >= 5:
            break
        print(f"\nModel for scipy.stats.{dist_name}: {model.__name__}")
        print("Required Parameters:")
        # Print fields and their types/defaults
        for name, field in model.model_fields.items():
             required_status = "Required (gt 0)" if field.is_required() else f"Default: {field.default}"
             print(f"- {name}: {field.annotation.__name__} ({required_status})")

    # --- Example Usage: Validate the Beta Distribution ---
    if 'beta' in models:
        BetaParams = models['beta']
        print("\n--- Example Validation: Beta Distribution (requires 'a' and 'b') ---")
        
        # 1. Valid data
        valid_data = {'a': 2.5, 'b': 7.0, 'loc': 10, 'scale': 5}
        try:
            validated_beta = BetaParams(**valid_data)
            print(f" Success: Validated Beta data: {validated_beta.model_dump_json(indent=2)}")
        except Exception as e:
            print(f" Error during valid validation: {e}")

        # 2. Invalid data (missing required 'a')
        invalid_data = {'b': 7.0, 'loc': 10, 'scale': 5}
        try:
            BetaParams(**invalid_data)
        except Exception as e:
            print(f"\n Error (as expected): Validation failed for missing 'a' parameter.")
            # Print the detailed validation error (Pydantic's ValidationError)
            print(e)
    
    # --- Example Usage: Validate the Normal Distribution ---
    if 'norm' in models:
        NormParams = models['norm']
        print("\n--- Example Validation: Normal Distribution (only 'loc' and 'scale') ---")
        
        # 3. Valid data for Normal distribution
        valid_norm_data = {'loc': 50, 'scale': 10}
        try:
            validated_norm = NormParams(**valid_norm_data)
            print(f" Success: Validated Normal data: {validated_norm.model_dump_json(indent=2)}")
        except Exception as e:
            print(f" Error during valid validation: {e}")
