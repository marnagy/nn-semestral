import pandas as pd
import os


def main():
    dirname = os.path.dirname(__file__)

    DailyStats = pd.read_csv(dirname+"/data-slovakia/DailyStats/OpenData_Slovakia_Covid_DailyStats.csv", sep=';')
    DailyStats['AgPosit'] = DailyStats['AgPosit'].fillna(0)
    DailyStats['AgTests'] = DailyStats['AgTests'].fillna(0)

    VaccReg = pd.read_csv(dirname+"/data-slovakia/Vaccination/OpenData_Slovakia_Vaccination_Regions.csv", sep=';')
    VaccReg = VaccReg.groupby('Date').sum()

    #print(DailyStats.set_index('Datum'))
    #print(VaccReg)

    joined = DailyStats.join(VaccReg, on='Datum')
    joined.fillna(0)
    joined['prirustkovy_pocet_nakazenych'] = joined['Dennych.PCR.prirastkov'] + joined['AgPosit']
    joined['all_doses'] = joined['first_dose'] + joined['second_dose'] + joined['third_dose']

    joined['first_vaccine_cumulative'] = joined['first_dose'].cumsum(skipna = True)
    joined['second_vaccine_cumulative'] = joined['second_dose'].cumsum(skipna = True)
    joined['third_vaccine_cumulative'] = joined['third_dose'].cumsum(skipna = True)
    joined['all_doses_cumulative'] = joined['all_doses'].cumsum(skipna = True)
    
    joined['kumulativni_pocet_nakazenych'] = joined['prirustkovy_pocet_nakazenych'].cumsum(skipna = True)
    
    joined['kumulativni_pocet_umrti'] = joined['Pocet.umrti'].cumsum(skipna = True)

    combined = joined.filter(['Datum','Dennych.PCR.testov','AgTests',
                                'prirustkovy_pocet_nakazenych', 'kumulativni_pocet_nakazenych',
                                #vyliecenych nemam
                                'Pocet.umrti', 'kumulativni_pocet_umrti',
                                'first_vaccine_cumulative',
                                'second_vaccine_cumulative',
                                'third_vaccine_cumulative',
                                'all_doses_cumulative',
                                'first_dose', 'second_dose', 'third_dose', 'all_doses'], axis=1)


    new_combined_columns = [
        'Datum',
        'Dennych.PCR.testov',
        'AgTests',
        'prirustkovy_pocet_nakazenych', # ->
        'kumulativni_pocet_nakazenych',
        'prirustkovy_pocet_umrti',
        'kumulativni_pocet_umrti',
        'first_vaccine_cumulative',
        'second_vaccine_cumulative',    # >|
        'third_vaccine_cumulative',
        'all_doses_cumulative',
        'first_dose',
        'second_dose',
        'third_dose',
        'all_doses'
    ]

    #print(list(combined.columns.values))

    # Set names to the columns
    keys = list(combined.columns.values)
    values = new_combined_columns
    combined = combined.rename(columns=dict(zip(keys, values)))

    #print(list(combined.columns.values))

    combined.to_csv(dirname+"\combined.csv", index=False)




if __name__ == '__main__':
    main()