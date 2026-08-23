//following rust tutorial for now before NN

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
