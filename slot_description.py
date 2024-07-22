RestaurantSlot = """

In the DOMAIN of "restaurant", the values that should be captured are:
 - "pricerange" that specifies the price range of the restaurant (cheap/moderate/expensive)
 - "area" that specifies the area where the restaurant is located (north/east/west/south/centre)
 - "food" that specifies the type of food the restaurant serves
 - "name" that specifies the name of the restaurant
 - "bookday" that specifies the day of the booking
 - "booktime" that specifies the time of the booking
 - "bookpeople" that specifies for how many people is the booking made
"""

HotelSlot = """
In the DOMAIN of "restaurant", the values that should be captured are:
 - "area" that specifies the area where the hotel is located (north/east/west/south/centre)
 - "internet" that specifies if the hotel has internet (yes/no)
 - "parking" that specifies if the hotel has parking (yes/no)
 - "stars" that specifies the number of stars the hotel has (1/2/3/4/5)
 - "type" that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - "pricerange" that specifies the price range of the hotel (cheap/expensive)
 - "name" that specifies name of the hotel
 - "bookstay" specifies length of the stay
 - "bookday" specifies the day of the booking
 - "bookpeople" specifies how many people should be booked for.
"""

TrainSlot = """
In the DOMAIN of "train", the values that should be captured are:
 - "arriveby" that specifies what time the train should arrive
 - "leaveat" that specifies what time the train should leave
 - "day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
 - "bookpeople" that specifies how many people the booking is for
"""

TaxiSlot = """
In the DOMAIN of "taxi", the values that should be captured are:
 - "arriveby" that specifies what time the train should arrive
 - "leaveat" that specifies what time the train should leave
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
"""

HospitalSlot = """
In the DOMAIN of "hospital", the values that should be captured are:
 - "department" that specifies the department of interest
"""

BusSlot = """
In the DOMAIN of "bus", the values that should be captured are:
 - "day" that specifies what day the bus should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "departure" that specifies the departure station
 - "destination" that specifies the destination station
"""

AttractionSlot = """
In the DOMAIN of "attraction", the values that should be captured are:
 - "type" that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - "area" that specifies the area where the attraction is located (north/east/west/south/centre)
 - "name" that specigies the name of the attraction
"""

DOMAIN_SLOT_DESCRIPTION = {
    "restaurant": RestaurantSlot,
    "hotel": HotelSlot,
    "train": TrainSlot,
    "taxi": TaxiSlot,
    "hospital": HospitalSlot,
    "bus": BusSlot,
    "attraction": AttractionSlot
}

DOMAIN_EXPECTED_SLOT = {
    "restaurant": ["pricerange", "area", "food", "name", "bookday", "booktime", "bookpeople"],
    "hotel": ["area", "internet", "parking", "stars", "type", "pricerange", "name", "bookday", "bookpeople", "bookstay"],
    "train": ["arriveby", "leaveat", "bookpeople", "day", "departure", "destination"],
    "taxi": ['departure', 'destination', 'leaveat', 'arriveby'],
    "hospital": ['department'],
    "bus": ["day", "departure", "destination", "leaveat"],
    "attraction": ["type", "area", "name"]
}

EXPECTED_DOMAIN = ["train", "attraction", "taxi", "hotel", "restaurant","bus", "hospital"]