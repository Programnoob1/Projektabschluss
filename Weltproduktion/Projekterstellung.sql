# Projekt anlegen

create database stahlprojekt;
use stahlprojekt;

#Tabellen anlegen

create table top_10produzenten (
	Land varchar(50),
    Produktion_Mio_Tonnen decimal(10,1)
);

# csv importieren mit "tableWizard"

create table verbauch_pro_kopf (
	Land varchar(50),
    Verbrauch_pro_kopf decimal(10,1)
    );

# csv importieren mit "tableWizard"

create table stahlhandel (
	Kategorie varchar(50),
    Menge_Mio_Tonnen decimal(10,1)
    );
drop table stahlhandel;
    # csv importieren mit "tableWizard"
    
create table Emissionen (
	Land varchar (50),
    Emissionen_t_pro_Tonne decimal(10,1)
    );
       # csv importieren mit "tableWizard"
 
    

