import matplotlib.pyplot as plt
import numpy as np

# ===============================================================
# METRICS (Resume + Medical) FOR ALL SIX METHODS
# ===============================================================

methods = [
    "Fewshot",
    "LoRA",
    "Curriculum",
    "Constraint",
    "Synthetic",
    "Hybrid"
]

# Resume Metrics
resume_exact =      [0.00, 58.06, 9.68, 9.68, 35.48, 48.39]
resume_name =       [91.67, 100.0, 93.55, 93.55, 96.77, 90.32]
resume_email =      [50.00, 90.32, 96.77, 96.77, 96.77, 93.55]
resume_skills =     [8.33, 58.06, 54.84, 74.19, 70.97, 81.61]
resume_experience = [16.67, 58.06, 83.87, 83.87, 48.39, 74.19]
resume_lev =        [70.00, 34.65, 788.87, 44.29, 37.06, 19.10]

# Medical Metrics
medical_exact =      [0.00, 46.67, 16.67, 16.67, 10.00, 43.33]
medical_name =       [94.12, 100.0, 76.67, 76.67, 100.0, 80.00]
medical_email =      [76.47, 96.67, 93.33, 93.33, 100.0, 86.67]
medical_skills =     [11.76, 70.00, 46.67, 70.00, 33.33, 76.61]
medical_experience = [17.65, 73.33, 90.00, 90.00, 56.67, 76.67]
medical_lev =        [354.00, 4.40, 890.27, 44.80, 43.27, 27.17]

# ===============================================================
# Helper: sort methods ascending except ALWAYS put Hybrid last
# ===============================================================

def sort_with_hybrid_last(values):
    indexed = list(enumerate(values))
    hybrid = indexed[-1]
    rest = indexed[:-1]
    rest_sorted = sorted(rest, key=lambda x: x[1])
    return rest_sorted + [hybrid]


# ===============================================================
# DEFAULT BAR PLOTS (Name, Email, Skills, Experience)
# Hybrid gets special color + dotted vertical guide line
# ===============================================================

def plot_metric(metric_name, resume_values, medical_values):
    sorted_resume = sort_with_hybrid_last(resume_values)
    sorted_med = sort_with_hybrid_last(medical_values)

    ordered_methods = [methods[idx] for idx, _ in sorted_resume]
    resume_ordered = [v for _, v in sorted_resume]
    medical_ordered = [v for _, v in sorted_med]

    x = np.arange(len(ordered_methods))
    width = 0.35

    plt.figure(figsize=(14, 6))

    for i, method in enumerate(ordered_methods):
        if method == "Hybrid":
            resume_color = "limegreen"
            medical_color = "gold"
        else:
            resume_color = "steelblue"
            medical_color = "darkorange"

        plt.bar(x[i] - width/2, resume_ordered[i], width,
                label="Resume" if i == 0 else "", color=resume_color)
        plt.bar(x[i] + width/2, medical_ordered[i], width,
                label="Medical" if i == 0 else "", color=medical_color)

        # Add dotted vertical guide for Hybrid
        if method == "Hybrid":
            hy_top = max(resume_ordered[i], medical_ordered[i])
            plt.plot([x[i], x[i]], [0, hy_top],
                     linestyle="dotted", color="black", linewidth=2,zorder=-1)

    plt.xticks(x, ordered_methods, rotation=45, fontsize=24)
    plt.ylabel(metric_name, fontsize=18)
    plt.title(f"{metric_name} Comparison Across Methods\n(Ascending Order, Hybrid Highlighted)", fontsize=20)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# ===============================================================
# SPECIAL GRAPH 1: EXACT MATCH = HORIZONTAL BAR CHART
# ===============================================================

def plot_exact_horizontal():
    sorted_resume = sort_with_hybrid_last(resume_exact)
    sorted_med = sort_with_hybrid_last(medical_exact)

    ordered_methods = [methods[idx] for idx, _ in sorted_resume]
    resume_ordered = [v for _, v in sorted_resume]
    medical_ordered = [v for _, v in sorted_med]

    y = np.arange(len(ordered_methods))

    plt.figure(figsize=(14, 6))

    for i, method in enumerate(ordered_methods):
        if method == "Hybrid":
            rcolor = "limegreen"
            mcolor = "gold"
        else:
            rcolor = "steelblue"
            mcolor = "darkorange"

        plt.barh(y[i] - 0.2, resume_ordered[i], 0.4, color=rcolor,
                 label="Resume" if i == 0 else "")
        plt.barh(y[i] + 0.2, medical_ordered[i], 0.4, color=mcolor,
                 label="Medical" if i == 0 else "")

        # dotted guide
        if method == "Hybrid":
            hy_top = max(resume_ordered[i], medical_ordered[i])
            plt.plot([0, hy_top], [y[i], y[i]], linestyle="dotted", color="black")

    plt.yticks(y, ordered_methods, fontsize=24)
    plt.xlabel("Exact Match (%)", fontsize=18)
    plt.title("Exact Match (Horizontal Ranked Plot)", fontsize=20)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# ===============================================================
# SPECIAL GRAPH 2: LEVENSHTEIN = LINE CHART WITH MARKERS
# ===============================================================

def plot_levenshtein_line():
    sorted_resume = sort_with_hybrid_last(resume_lev)
    sorted_med = sort_with_hybrid_last(medical_lev)

    ordered_methods = [methods[idx] for idx, _ in sorted_resume]
    resume_ordered = [v for _, v in sorted_resume]
    medical_ordered = [v for _, v in sorted_med]

    x = np.arange(len(ordered_methods))

    plt.figure(figsize=(14, 6))

    plt.plot(x, resume_ordered, marker="o", color="steelblue", linewidth=2, label="Resume")
    plt.plot(x, medical_ordered, marker="o", color="darkorange", linewidth=2, label="Medical")

    # Highlight Hybrid point with special marker + dotted guide
    hy_idx = ordered_methods.index("Hybrid")
    hy_r = resume_ordered[hy_idx]
    hy_m = medical_ordered[hy_idx]

    plt.scatter(hy_idx, hy_r, color="limegreen", s=120, edgecolor="black", label="Hybrid Resume")
    plt.scatter(hy_idx, hy_m, color="gold", s=120, edgecolor="black", label="Hybrid Medical")

    plt.plot([hy_idx, hy_idx], [0, max(hy_r, hy_m)], linestyle="dotted", color="black")

    plt.xticks(x, ordered_methods, rotation=45, fontsize=24)
    plt.ylabel("Levenshtein Distance", fontsize=18)
    plt.title("Levenshtein Distance Across Methods (Line Chart)", fontsize=20)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# ===============================================================
# GENERATE ALL GRAPHS
# ===============================================================

plot_exact_horizontal()
plot_metric("Name Accuracy (%)", resume_name, medical_name)
plot_metric("Email Accuracy (%)", resume_email, medical_email)
plot_metric("Skills Accuracy (%)", resume_skills, medical_skills)
plot_metric("Experience Accuracy (%)", resume_experience, medical_experience)
plot_levenshtein_line()

print("All graphs generated successfully (with Hybrid highlighted).")
