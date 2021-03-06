
SET foreign_key_checks = 0;

SELECT * from MOVIEDATAMODEL.MOVIE;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/data.csv"
INTO TABLE MOVIEDATAMODEL.MOVIE
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/person.csv"
INTO TABLE MOVIEDATAMODEL.PERSON
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/short.csv"
INTO TABLE MOVIEDATAMODEL.SHORT
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/title.csv"
INTO TABLE MOVIEDATAMODEL.TITLE
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/tvMovie.csv"
INTO TABLE MOVIEDATAMODEL.TV_MOVIE
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/votes.csv"
INTO TABLE MOVIEDATAMODEL.VOTES
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/genre.csv"
INTO TABLE MOVIEDATAMODEL.GENRE
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/director.csv"
INTO TABLE MOVIEDATAMODEL.DIRECTOR
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/producer.csv"
INTO TABLE MOVIEDATAMODEL.PRODUCER
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/writer.csv"
INTO TABLE MOVIEDATAMODEL.WRITER
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/actor.csv"
INTO TABLE MOVIEDATAMODEL.ACTOR
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

LOAD DATA LOCAL INFILE "C/Users/Keinan/Documents/College_Classes/Spring_2017/Databases/data/tv.csv"
INTO TABLE MOVIEDATAMODEL.TV_SERIES
CHARACTER SET latin1
FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
LINES TERMINATED BY '\r\n' STARTING BY ''
IGNORE 1 LINES
;

select * from MovieDataModel.Movie;

