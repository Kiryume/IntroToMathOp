def solve_kkt_min(f, vars, g_list, show_latex=False):
    """
    Solves a minimization problem using KKT conditions.
    
    Args:
        f: The symbolic function to minimize.
        vars: A list of the variables (e.g., [x, y]).
        g_list: A list of functions defining the constraints g_i <= 0.
        
    Returns:
        Prints the KKT points and the optimal solution.
    """
    # 1. Define Lagrange multipliers (one for each constraint)
    # Using l_0, l_1, etc.
    lambdas = [var(f'lambda_{i}') for i in range(len(g_list))]
    
    print(f"Objective Function: {f}")
    print(f"Constraints (<= 0): {g_list}")
    
    # 2. Construct KKT Equations
    
    # A) Gradient of Lagrangian = 0
    # L = f + sum(lambda_i * g_i)
    # dL/dx = df/dx + sum(lambda_i * dg_i/dx) = 0
    grad_eqs = []
    for v in vars:
        eq = diff(f, v) + sum(lambdas[i] * diff(g_list[i], v) for i in range(len(g_list))) == 0
        grad_eqs.append(eq)
        
    # B) Complementary Slackness
    # lambda_i * g_i = 0
    slack_eqs = []
    for i in range(len(g_list)):
        eq = lambdas[i] * g_list[i] == 0
        slack_eqs.append(eq)
        
    # Combine all equations to solve
    all_eqs = slack_eqs + grad_eqs
    all_vars = vars + lambdas
    print('The KKT conditions are')
    for l in lambdas:
        if show_latex:
            print(latex(l>=0))
        else:
            show(l>=0)
    for g in g_list:
        if show_latex:
            print(latex(g<=0))
        else:
            show(g<=0)
    for eq in all_eqs:
        if show_latex:
            print(latex(eq))
        else:
            show(eq)
            
    
    print("\nSolving system of KKT equations...")
    try:
        solutions = solve(all_eqs, all_vars, solution_dict=True)
        normalized_solutions = []
        for sol in solutions:
            new_sol = sol.copy()
            for var_needed in all_vars:
                # If a required variable is missing from keys...
                if var_needed not in new_sol:
                    # ...scan the values to see if it's there
                    for k, v in sol.items():
                        if v == var_needed:
                            # Found it! Flip it: set variable = constant
                            new_sol[var_needed] = k
                            break
            normalized_solutions.append(new_sol)
        solutions = normalized_solutions
    except Exception as e:
        print(f"Error solving system: {e}")
        return

    valid_candidates = []

    # 3. Filter Solutions
    for sol in solutions:
        try:
            # Extract numerical values for variables and lambdas
            # using .n() to handle symbolic constants like sqrt(2)
            pt = [sol[v].n() for v in vars]
            l_vals = [sol[l].n() for l in lambdas]
            
            # Check 1: Real values only
            if not all(val.is_real() for val in pt + l_vals):
                continue
                
            # Check 2: Primal Feasibility (g_i(x) <= 0)
            # We use a small tolerance for floating point comparisons
            tol = 1e-6
            primal_feasible = True
            broken_g = {}
            for g in g_list:
                val = g.subs(sol).n()
                if val > tol:
                    primal_feasible = False
                    break
            
            # Check 3: Dual Feasibility (lambda_i >= 0)
            dual_feasible = True
            broken_lval = {}
            for i,l_val in enumerate(l_vals):
                if l_val < -tol:
                    dual_feasible = False
                    break
            
            if primal_feasible and dual_feasible:
                f_val = f.subs(sol).n()
                valid_candidates.append({
                    'point': sol,
                    'f_val': f_val,
                    'type': 'Valid KKT Point'
                })
                
        except (TypeError, ValueError, AttributeError):
            # Skip solutions that cannot be evaluated numerically (e.g. complex symbolic)
            continue

    # 4. Find Minimum
    if not valid_candidates:
        print("No valid KKT points found.")
    else:
        # Sort candidates by function value
        valid_candidates.sort(key=lambda x: x['f_val'])
        
        print(f"\nFound {len(valid_candidates)} valid candidate(s).")
        best = valid_candidates[0]
        
        print("-" * 30)
        print("OPTIMAL SOLUTION FOUND:")
        print(f"Minimum Value: {best['f_val']}")
        print("At Point:")
        for v in vars:
            print(f"  {v} = {best['point'][v]}")
        print("-" * 30)

