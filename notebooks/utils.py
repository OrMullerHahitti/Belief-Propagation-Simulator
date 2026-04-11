import matplotlib.pyplot as plt
import numpy as np

def get_lower_envelope(slopes, intercepts):
    """
    Calculates the lower envelope (minimum) of a set of lines y = m*x + c.
    Returns a list of segments: [{'start': x1, 'end': x2, 'val': active_slope}, ...]
    """
    lines = sorted(zip(slopes, intercepts), key=lambda x: (x[0], x[1]), reverse=True)
    
    unique_lines = []
    if lines:
        unique_lines.append(lines[0])
        for i in range(1, len(lines)):
            if lines[i][0] == lines[i-1][0]:
                continue 
            unique_lines.append(lines[i])
    
    stack = [] 
    for m_curr, c_curr in unique_lines:
        while len(stack) > 0:
            start_prev, m_prev, c_prev = stack[-1]
            x_int = (c_curr - c_prev) / (m_prev - m_curr)
            if x_int <= start_prev:
                stack.pop()
            else:
                stack.append((x_int, m_curr, c_curr))
                break
        if not stack:
            stack.append((-np.inf, m_curr, c_curr))
            
    intervals = []
    for i in range(len(stack)):
        start = stack[i][0]
        end = stack[i+1][0] if i + 1 < len(stack) else np.inf
        intervals.append({'start': start, 'end': end, 'val': stack[i][1]})
    return intervals

def plot_gap_preserved_regions(cost_matrix, axis=1, edge_values=None, delta_q_range=(-15, 15), title=None):
    if isinstance(cost_matrix, list):
        M = np.array(cost_matrix)
    else:
        M = cost_matrix
    
    if axis == 0:
        M = M.T
    
    n_out, n_in = M.shape
    
    if edge_values is None:
        edge_values = np.arange(n_in)
    else:
        edge_values = np.array(edge_values)

    # 1. Get exact active intervals
    all_row_intervals = []
    all_breakpoints = set([-np.inf, np.inf])
    
    for i in range(n_out):
        row_intervals = get_lower_envelope(edge_values, M[i])
        all_row_intervals.append(row_intervals)
        for interval in row_intervals:
            all_breakpoints.add(interval['start'])
            all_breakpoints.add(interval['end'])

    # 2. Check segments for gap preservation
    sorted_breaks = sorted([x for x in all_breakpoints if x > -np.inf and x < np.inf])
    test_points = [-np.inf] + sorted_breaks + [np.inf]
    valid_regions = []

    for k in range(len(test_points) - 1):
        x_start, x_end = test_points[k], test_points[k+1]
        
        if x_start == -np.inf: midpoint = x_end - 1.0
        elif x_end == np.inf: midpoint = x_start + 1.0
        else: midpoint = (x_start + x_end) / 2.0
        
        active_values = []
        for i in range(n_out):
            for interval in all_row_intervals[i]:
                if interval['start'] <= midpoint <= interval['end']:
                    active_values.append(interval['val'])
                    break
        
        if len(set(active_values)) == 1:
            vis_start = max(x_start, delta_q_range[0])
            vis_end = min(x_end, delta_q_range[1])
            if vis_end > vis_start:
                valid_regions.append((vis_start, vis_end))

    # ================= PLOTTING =================
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_out))
    
    # 3. Pre-calculate curves for Global Min detection
    x_grid = np.linspace(delta_q_range[0], delta_q_range[1], 2000)
    all_envelopes = []
    
    for i in range(n_out):
        row_costs = M[i, :, np.newaxis] + np.outer(edge_values, x_grid)
        envelope = np.min(row_costs, axis=0)
        all_envelopes.append(envelope)
        
    all_envelopes = np.array(all_envelopes)
    # Find the global minimum across all output rows at every x
    global_min_curve = np.min(all_envelopes, axis=0)

    # 4. Plot Regions and Lines
    
    # Draw Green "Gap Preserved" Regions
    for (v_start, v_end) in valid_regions:
        plt.axvspan(v_start, v_end, color='green', alpha=0.1, lw=0)
        if (v_end - v_start) > (delta_q_range[1] - delta_q_range[0]) * 0.05:
            plt.text((v_start + v_end)/2, plt.ylim()[1], "LOCKED", 
                     color='green', ha='center', va='top', fontsize=8, fontweight='bold')

    for i in range(n_out):
        row_curve = all_envelopes[i]
        color = colors[i]
        letter = chr(97 + i)
        
        # A. Faded Line (Context)
        # Shows the trajectory of this variable even when it's not the winner
        plt.plot(x_grid, row_curve, color=color, alpha=0.4, linestyle='--', linewidth=1)
        
        # B. Solid Line (Global Winner)
        # Only plot this line solid where it is equal to the global minimum
        is_winner = np.isclose(row_curve, global_min_curve, atol=1e-5)
        masked_curve = np.ma.masked_where(~is_winner, row_curve)
        
        plt.plot(x_grid, masked_curve, color=color, label=f'{letter}', linewidth=3)
        
        # C. Intersection Points (Breakpoints)
        for interval in all_row_intervals[i]:
            bp = interval['start']
            
            # Only plot if inside range
            if delta_q_range[0] < bp < delta_q_range[1]:
                # Calculate y-value exactly
                y_val = np.min(M[i, :] + edge_values * bp)
                
                # Plot the dot
                plt.plot(bp, y_val, 'o', color=color, markersize=5, markeredgecolor='white')
                
                # Add Text Annotation with arrow
                # Stagger text slightly based on index to avoid overlap
                y_offset = (i % 2) * 2 + 1  
                
                plt.annotate(f"{bp:.2f}", 
                             xy=(bp, y_val), 
                             xytext=(bp, y_val + y_offset),
                             color=color, fontsize=9, fontweight='bold',
                             arrowprops=dict(arrowstyle="-", color=color, alpha=0.3),
                             ha='center')

    final_title = title if title else "Min-Sum Global Envelope & Switches"
    plt.title(final_title, fontsize=14)
    plt.xlabel(r"$\Delta Q$", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.xlim(delta_q_range)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Variable (Winner)", loc='upper right')
    plt.tight_layout()
    plt.show()

