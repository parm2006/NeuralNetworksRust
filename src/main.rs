//following rust tutorial for now before NN

fn loops() {
    //loops and conditionals
    let is_even = true;

    if is_even {
        println!("This is even!");
    } else if !is_even {
        println!("This is not even");
    } else {
        println!()
    }

    // for i in 0..100{
    //     print!("{}",i);
    // }

    let sentence: String = String::from("my name is parth");
    let firstword = get_first_word(sentence);
    println!("The first word is: {}", firstword);

    let n = 50;
    for _i in 0..n {
        println!("Hello dude");
    }
}

fn get_first_word(sentence: String) -> String {
    //you must define the return type of the function whenever you create it.
    let mut ans: String = String::from("");
    for char in sentence.chars() {
        if char == ' ' {
            break;
        }
        ans.push_str(char.to_string().as_str());
    }
    return ans;
}

fn strings() {
    let x: &str = "strisngs";

    //not a fixed type cuz there are not fixed length like vector so its hard to keep it
    //since it can change at runtime,

    //string mutation is hard for rust because you have to change the allocated space, not automatica, unlike nums which have fixed space
    //instead use string::from("...")
    let mut words: String = String::from("hello ");

    let greet = String::from("hello world");

    println!("{}", greet);

    let char1: Option<char> = greet.chars().nth(0); //so the type is optionally a character, or it will be none, so rust wont let us directly print this char, we have to use match/switch for the case if it is and hte case if its not.

    match char1 {
        //pattern matching
        Some(c) => print!("{}", c),
        None => print!("No char at this location"),
    }

    print!("char1:{}", char1.unwrap()) //unwrap will unwrap the option type and so we will be chill. it will cause a runtime exception
}

fn bool() {
    let mut is_male = false;

    //let isnotmale = !is_male;

    if is_male {
        print!("You are male");
    } else {
        is_male = true;
    }
    if is_male {
        print!("Now you are male")
    }
}

fn vars() {
    //simple variables in rust
    //define vars with let
    let x = 5;
    //let is the init-er
    let w: i32 = 6; //we can also declare the type of the variable we want to make it. //these are immutable by default

    //    let z:u8 = -4; //error because unsigned cant be negative
    let y: u32 = 1000;

    let f: f32 = 1.001; // float

    print!("x :{}", f);

    //runtime overflow: cant be checked by compiler obviously
}
