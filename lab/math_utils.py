def print_latex(*args, **kwargs):
    print(latex(args,kwargs))

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

def symmetric_reduction(A, show_latex=False):
    """
    Performs the symmetric reduction algorithm described in section 8.7.
    Transforms a symmetric matrix A into a diagonal matrix D using congruence transformations.
    Returns matrices B and D such that B.T * A * B = D.
    
    Args:
        A: A symmetric matrix (Sage matrix object)
    """
    # Check for symmetry
    if not A.is_symmetric():
        print("Error: Matrix is not symmetric.")
        return

    n = A.nrows()
    # Create the extended matrix (2n x n)
    # The top n rows form the identity matrix
    # The bottom n rows form the matrix A
    # We use QQ (Rational Field) to handle exact arithmetic
    M = matrix(QQ, 2*n, n)
    M[0:n, 0:n] = identity_matrix(n)
    M[n:2*n, 0:n] = A

    print("Initial extended matrix [I // A]:")
    if show_latex:
        print_latex(M)
    else:
        show(M)
    print("\n" + "="*40 + "\n")

    # Iterate through diagonal elements (pivots)
    for k in range(n):
        print(f"--- Processing pivot step k={k} ---")
        
        # Step 1: Ensure we have a non-zero pivot at M[n+k, k]
        # This corresponds to A[k, k]
        pivot_val = M[n+k, k]
        
        if pivot_val == 0:
            # Strategy A: Look for a non-zero diagonal element later in the matrix
            swap_index = -1
            for j in range(k+1, n):
                if M[n+j, j] != 0:
                    swap_index = j
                    break
            
            if swap_index != -1:
                print(f"Pivot A[{k},{k}] is zero.")
                print(f"Strategy: Swap index {k} with {swap_index} (non-zero diagonal found).")
                print(f"  1. Swap Column {k} <-> Column {swap_index} (Full matrix)")
                print(f"  2. Swap Row {k} <-> Row {swap_index} (Bottom A-block only)")
                
                M.swap_columns(k, swap_index)
                M.swap_rows(n+k, n+swap_index)
                
                print("\nMatrix after swap:")
                if show_latex:
                    print_latex(M)
                else:
                    show(M)
                print("-" * 20)
            else:
                # Strategy B: All subsequent diagonals are zero.
                # Look for a non-zero off-diagonal element A[j, k] with j > k
                add_index = -1
                for j in range(k+1, n):
                    if M[n+j, k] != 0:
                        add_index = j
                        break
                
                if add_index != -1:
                    print(f"Pivot A[{k},{k}] is zero and no non-zero diagonals exist.")
                    print(f"Strategy: Add index {add_index} to {k} (non-zero off-diagonal found).")
                    print(f"  1. Col {k} = Col {k} + Col {add_index}")
                    print(f"  2. Row {k} = Row {k} + Row {add_index} (Bottom A-block only)")
                    
                    M.add_multiple_of_column(k, add_index, 1)
                    M.add_multiple_of_row(n+k, n+add_index, 1)
                    
                    print("\nMatrix after addition:")
                    if show_latex:
                        print_latex(M)
                    else:
                        show(M)
                    print("-" * 20)
                else:
                    print(f"Column {k} in A block is strictly zero. No operations needed.")
                    continue

        # Update pivot after potential swaps/adds
        pivot = M[n+k, k]
        if pivot == 0:
            continue # Should be handled by logic above, but safety check for singular matrices

        # Step 2: Eliminate entries to the right of the pivot in the current row of A
        # We perform column operations to zero out A[k, j] for j > k
        # followed by symmetric row operations.
        for j in range(k+1, n):
            target = M[n+k, j] # Element at A[k, j]
            
            if target != 0:
                factor = target / pivot
                print(f"Eliminating A[{k}, {j}] = {target} using pivot {pivot}.")
                print(f"  factor = {target} / {pivot} = {factor}")
                print(f"  1. Col {j} = Col {j} - ({factor}) * Col {k}")
                print(f"  2. Row {j} = Row {j} - ({factor}) * Row {k} (Bottom A-block only)")
                
                # Column operation on full matrix
                M.add_multiple_of_column(j, k, -factor)
                # Row operation on bottom block (rows n to 2n-1)
                M.add_multiple_of_row(n+j, n+k, -factor)
                
                print("\nMatrix after elimination step:")
                if show_latex:
                    print_latex(M)
                else:
                    show(M)
                print("-" * 20)
        print("\n")

    # Extract result matrices
    B = M[0:n, 0:n]
    D = M[n:2*n, 0:n]
    
    print("="*40)
    print("Final Result:")
    print("Matrix B (Transformation):")
    print(B)
    print("\nMatrix D (Diagonal):")
    print(D)
    
    # Verification step
    # print("\nVerification (B.T * A * B == D):")
    # print(B.T * A * B == D)
    
    return B, D


def fourier_motzkin_step(inequalities, var_to_eliminate):
    """
    Performs one step of Fourier-Motzkin elimination for a single variable.
    
    Args:
        inequalities: A list of symbolic inequalities (e.g., [x + y <= 1, ...])
        var_to_eliminate: The symbolic variable to eliminate (e.g., z)
        
    Returns:
        A new list of inequalities with the variable eliminated.
    """
    upper_bounds = []
    lower_bounds = []
    independent = []
    
    print(f"--- Eliminating variable: {var_to_eliminate} ---")
    
    for ineq in inequalities:
        # Standardize inequality to form: expr <= 0
        # lhs() returns the left-hand side, rhs() the right.
        expr = (ineq.lhs() - ineq.rhs()).expand()
        
        # Get coefficient of the variable
        coeff = expr.coefficient(var_to_eliminate)
        
        # If coeff is 0, the inequality doesn't involve the variable
        if coeff == 0:
            independent.append(ineq)
            continue
            
        # Rearrange to isolate variable: 
        # If coeff > 0:  coeff * var + rest <= 0  -->  var <= -rest / coeff  (Upper Bound)
        # If coeff < 0:  coeff * var + rest <= 0  -->  var >= -rest / coeff  (Lower Bound)
        
        rest = expr - coeff * var_to_eliminate
        bound = -rest / coeff
        
        if coeff > 0:
            upper_bounds.append(bound)
            print(f"Found Upper Bound: {var_to_eliminate} <= {bound}")
        else:
            lower_bounds.append(bound)
            print(f"Found Lower Bound: {var_to_eliminate} >= {bound}")

    new_inequalities = []
    
    # 1. Combine every Lower Bound with every Upper Bound
    # L <= var <= U  implies  L <= U
    print(f"\nGenerating new constraints (L <= U):")
    if not lower_bounds and not upper_bounds:
         print("No bounds found for this variable. It is unconstrained.")
    elif not lower_bounds:
         print("Only upper bounds found. Variable can go to -infinity.")
    elif not upper_bounds:
         print("Only lower bounds found. Variable can go to +infinity.")
            
    for l in lower_bounds:
        for u in upper_bounds:
            # Form the new inequality: L <= U  -->  L - U <= 0
            new_ineq = (l <= u)
            print(f"  ({l}) <= ({u})  -->  {new_ineq}")
            new_inequalities.append(new_ineq)
            
    # 2. Add independent inequalities back to the set
    for indep in independent:
        new_inequalities.append(indep)
        
    print(f"--- Step Complete. New system size: {len(new_inequalities)} ---\n")
    return new_inequalities

def solve_fourier_motzkin(inequalities, vars_to_elim):
    """
    Iteratively eliminates a list of variables.
    """
    current_system = inequalities
    for var in vars_to_elim:
        current_system = fourier_motzkin_step(current_system, var)
    
    return current_system