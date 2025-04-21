import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # not used directly here, but imported for consistency
from scipy.stats import ttest_ind
# Load dataset (if needed)
df = pd.read_csv("C:\\Users\\mohit\\Desktop\\School_Building_Technology_Survey_20250412 (5).csv")
print(df)

# === Basic Data Exploration ===
print("\n--- BASIC DATA INSPECTION ---")
print("Head:\n", df.head())
print("\nTail:\n", df.tail())
print("\nInfo:")
df.info()
print("\nDescribe:\n", df.describe())

# Drop NA rows if needed
df_cleaned = df.dropna()
print("\nData shape after dropping NA:", df_cleaned.shape)

# === Statistical Measures ===
print("\n--- STATISTICAL MEASURES ---")
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    print(f"\nColumn: {col}")
    print("Mean:", df[col].mean())
    print("Std Dev:", df[col].std())
    print("Variance:", df[col].var())

col = 'number_of_instructional_laptops'

# Drop missing values for accurate calculation
data = df[col].dropna()

# Calculate IQR bounds
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

# Output
print(f"\n--- OUTLIER DETECTION for '{col}' ---")
print("Q1:", Q1)
print("Q3:", Q3)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Number of Outliers:", outliers.shape[0])
outliers[[col]]


# Objective 1: Visualize the distribution of wireless internet coverage across schools
cols = [
    'one_classroom_or_meeting_room_only',
    'more_than_one_classroom_or_meeting_room_but_not_buildingwide',
    'central_areas_throughout_building_but_not_one_wapclassroom',
    'the_entire_building'
]

coverage_sums = df[cols].sum()

plt.figure(figsize=(10, 5))
sns.barplot(x=coverage_sums.index, y=coverage_sums.values)
plt.title("Wireless Internet Coverage Across Schools")
plt.ylabel("Number of Schools")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Objective 2: Analyze relationship between student enrollment and number of instructional laptops

# Scatterplot of enrollment vs. number of laptops
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="k12_total_enrollment",
    y="number_of_instructional_laptops"
)
plt.title("Enrollment vs Instructional Laptops")
plt.xlabel("K-12 Enrollment")
plt.ylabel("Number of Laptops")
plt.tight_layout()
plt.show()

# Objective 3: Show percentage distribution of different operating systems used in schools

os_cols = [
    'number_of_windows_10_os', 'number_of_windows_7_os', 'number_of_linux_os',
    'number_of_mac_os_x_105_or_higher', 'number_of_google_chrome', 'number_of_ios',
    'number_of_android'
]

os_totals = df[os_cols].sum()

plt.figure(figsize=(8, 8))
plt.pie(os_totals, labels=os_cols, autopct='%1.1f%%', startangle=140)
plt.title("Operating System Distribution in Schools")
plt.tight_layout()
plt.show()

# Objective 4: Compare student enrollment based on presence of 1-to-1 device initiatives

# Create column indicating presence of 1-to-1 program
df['has_1to1'] = df[
    'do_you_have_one_or_more_grades_in_your_school_that_have_a_onetoone_initiative_eg_laptops_tablets_netbooks'
]

# Boxplot comparing enrollment
plt.figure(figsize=(8, 6))
sns.boxplot(x="has_1to1", y="k12_total_enrollment", data=df)
plt.title("Enrollment by 1-to-1 Device Initiative Presence")
plt.xlabel("Has 1-to-1 Initiative")
plt.ylabel("K-12 Enrollment")
plt.tight_layout()
plt.show()

# Objective 5: Test if schools with full internet access have significantly more laptops (t-test)

# Define groups based on full vs partial internet access
full_internet = df['number_of_classrooms_including_portables_used_as_classrooms_labs_and_library'] == \
                df['number_of_classrooms_with_wired_or_wireless_internet_access']

group_full = df[full_internet]['number_of_instructional_laptops']
group_partial = df[~full_internet]['number_of_instructional_laptops']

# Perform independent t-test
t_stat, p_val = ttest_ind(group_full, group_partial, equal_var=False)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_val:.4f}")

# Load the dataset
file_path = 'cleaned_school_tech_survey_python_ready.csv'
df = pd.read_csv(file_path)

# Select relevant columns for analysis
selected_columns = [
    'k12_total_enrollment',
    'number_of_instructional_laptops',
    'number_of_instructional_desktops',
    'number_of_multimedia_projectors',
    'devices_with_ram_1_gb_or_higher'
]

# Drop rows with missing values in selected columns
df_clean = df[selected_columns].dropna()

# Line Graph: Enrollment vs Device Counts
plt.figure(figsize=(10, 5))
for col in selected_columns[1:]:
    plt.plot(df_clean['k12_total_enrollment'], df_clean[col], label=col)
plt.xlabel('K-12 Total Enrollment')
plt.ylabel('Device Count')
plt.title('Enrollment vs Device Count (Line Graph)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Box Plot: Distribution of Device Counts
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean[selected_columns[1:]])
plt.title('Device Counts Distribution (Box Plot)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap: Correlation Matrix
plt.figure(figsize=(8, 6))
corr = df_clean.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Barplot: Average Devices per Type
avg_values = df_clean[selected_columns[1:]].mean()
plt.figure(figsize=(8, 6))
sns.barplot(x=avg_values.index, y=avg_values.values)
plt.title('Average Number of Devices per School (Barplot)')
plt.ylabel('Average Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatterplot: Laptops vs Desktops
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='number_of_instructional_laptops',
    y='number_of_instructional_desktops',
    data=df_clean
)
plt.title('Laptops vs Desktops (Scatterplot)')
plt.xlabel('Instructional Laptops')
plt.ylabel('Instructional Desktops')
plt.grid(True)
plt.tight_layout()
plt.show()