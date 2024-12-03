import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import kagglehub
import zipfile
import datetime
import geopandas as gpd


def scrape_web():
    with zipfile.ZipFile("hurricane-database.zip", 'r') as zip_ref:
        zip_ref.extractall()

    path = kagglehub.dataset_download("noaa/hurricane-database")


def clean_atlantic_data():
    """
    This function changes the dates into datetime objects, and drops the
    "Event" column.
    :return:
    """
    df = pd.read_csv("atlantic.csv")
    new_dates = []

    # Changes dates to datetime objects
    for date in df["Date"]:
        date = str(date)
        year = date[0:4]
        month = date[4:6]
        day = date[6:]
        new_dates.append(datetime.date(int(year), int(month), int(day)))

    df["Date"] = new_dates

    df.drop("Event", axis=1, inplace=True)
    return df


def wind_pressure(df):
    df = df[df["Maximum Wind"] >= 0]
    df = df[df["Minimum Pressure"] >= 0]

    X = df["Minimum Pressure"]
    y = df["Maximum Wind"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)

    fig, ax = plt.subplots()

    ax.scatter(df["Minimum Pressure"], df["Maximum Wind"], color='darkorange')

    ax.plot(df["Minimum Pressure"], y_pred, color='teal')

    ax.set_facecolor("floralwhite")
    ax.set_xlabel("Minimum Pressure (mb)", fontsize=16, fontname="Lucida Sans Unicode")
    ax.set_ylabel("Maximum Wind Speed (mph)", fontsize=16, fontname="Lucida Sans Unicode")
    ax.set_xlim(880, 1026)
    ax.set_title("Wind Pressure and Speeds", fontsize=20, fontname="Lucida Sans Unicode")
    fig.patch.set_facecolor("whitesmoke")


def lats_longs_anom(df, direction):
    new_direction = []

    for i in range(len(df)):
        cur = df.iloc[i][direction]
        new_cur = cur[:-1]
        cur_neg = float(new_cur) * -1

        if "W" in cur or "S" in cur:
            new_direction.append(float(cur_neg))
        else:
            new_direction.append(float(new_cur))

    return new_direction


def five_years(df, start, end):
    new_vals = []

    for index, row in df.iterrows():
        if start <= row["Date"].year <= end:
            new_vals.append(row)

    return pd.DataFrame(new_vals)


def cyclone_path(df, cname=None):
    new_long = lats_longs_anom(df, "Longitude")
    new_lat = lats_longs_anom(df, "Latitude")

    df["Longitude"] = new_long
    df["Latitude"] = new_lat

    g_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world_data = gpd.read_file(url)

    continents = ["North America", "South America", "Africa", "Europe"]
    new_wrld_data = world_data[world_data["CONTINENT"].isin(continents)]

    fig, axis = plt.subplots()
    new_wrld_data.plot(ax=axis, color="lightblue", edgecolor="whitesmoke")

    g_df.plot(ax=axis, color="darkorange", alpha=0.2)
    fig.patch.set_facecolor("whitesmoke")
    axis.set_facecolor("floralwhite")
    axis.set_xlabel("Longitude", fontsize=16, fontname="Lucida Sans Unicode")
    axis.set_ylabel("Latitude", fontsize=16, fontname="Lucida Sans Unicode")
    axis.set_xlim(-150, 50)

    if cname:
        title_str = "Cyclone Path (" + cname + ")"
        axis.set_title(title_str, fontsize=20, fontname="Lucida Sans Unicode")
    else:
        axis.set_title("Cyclone Path", fontsize=20, fontname="Lucida Sans Unicode")


def cyclones_over_time(df):
    """
    cyclones yearly
    :param df:
    :return:
    """
    date_at = {}

    for index, row in df.iterrows():
        if row["Date"].year not in date_at:
            date_at[row["Date"].year] = 1
        else:
            date_at[row["Date"].year] += 1

    x_years = list(range(1851, 2016, 4))
    y_ticks = list(range(0, 950, 50))

    # Create a figure and set of subplots
    fig, ax = plt.subplots()
    ax.set_facecolor("floralwhite")
    plt.plot(list(date_at.keys()), list(date_at.values()), color="darkorange")
    plt.xticks(ticks=x_years, rotation=45)
    plt.yticks(y_ticks)
    ax.set_xlabel("Year", fontsize=16, fontname="Lucida Sans Unicode")
    ax.set_ylabel("Frequency", fontsize=16, fontname="Lucida Sans Unicode")
    ax.set_xlim(1851, 2015)
    ax.set_title("Frequency of Tropical Cyclones by Year", fontsize=20, fontname="Lucida Sans Unicode")
    fig.patch.set_facecolor("whitesmoke")


def main():
    scrape_web()
    df = clean_atlantic_data()
    wind_pressure(df)

    katrina_data = five_years(df, 2005, 2005)
    humberto_data = five_years(df, 2007, 2007)
    cyclone_path(katrina_data, "Katrina")
    cyclone_path(humberto_data, "Humberto")

    cyclones_over_time(df)

    plt.show()


if __name__ == '__main__':
    main()
